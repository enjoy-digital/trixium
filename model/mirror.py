#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Python mirrors of the firmware's inference loops, the references the simulation and the card
are compared against (test/system/test_sim_smoke.py, test/firmware/test_hybrid.py,
software/lab/mirror.py).

hybrid_generate mirrors firmware/infer.c (the resident-BFP loop of the dense models):
engine-bitexact matvecs, int16 scratchpad-resident activations with integer exponent
algebra, float32 CPU op math, per-op cpu/hw switches (hw ops run the vector-unit microcode
interpreter).

hybrid27_generate mirrors firmware/hybrid.c (the hybrid models' CPU path): it walks a BONS
v3 image, every matvec through the engine's bit-exact model (int8 input with a power-of-two
scale, int32 results, float scaling), with sequential float32 sums for the other operations.
The default SiLU models the RV32F baseline's target-libm LUT, fused interpolation and
polynomial tails; exact=True retains direct mathematical SiLU for comparison with the FP32
golden. Other transcendental operations still use NumPy; passing the pinned tiny trace does
not prove bit-exactness for arbitrary models, inputs or target libm implementations.
"""

import numpy as np

from model import layout
from model import bitexact as bx

f32 = np.float32

# Dense Mirror -------------------------------------------------------------------------------------

def hybrid_generate(img, prompt, n_new, modes=None, trace=None):
    """Python mirror of the firmware loop (resident-BFP): engine-bitexact matvecs,
    int16 scratchpad-resident activations with integer exponent algebra, float32 CPU op
    math mirroring the soft-float firmware, per-op cpu/hw switches (hw ops run the actual
    microcode interpreter). modes: dict op->"cpu"|"hw" (default all-cpu)."""
    from model import asm as vasm
    h    = img.header
    f32  = np.float32
    E_NORM = -11

    if modes is None:
        modes = {}

    def lroundf(x):
        return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int64)

    def fsum32(v):
        acc = f32(0.0)
        for e in np.asarray(v, dtype=f32):
            acc = f32(acc + e)
        return acc

    def entry(name):
        return img.by_name[name]

    def tensor(name):
        return img.tensor(name)

    def norm_w(name):
        _, v = tensor(name)
        return v.reshape(-1).astype(f32)

    def to_bfp16(x):
        amax = np.abs(x).max()
        if amax == 0:
            return np.zeros(len(x), dtype=np.int64), 0
        e = int(np.frexp(f32(amax))[1]) - 15
        return np.clip(lroundf(x*f32(np.exp2(-e))), -32767, 32767), e

    def w16_exp(w16_name):
        # The .w16 entry's bias byte is the tensor's exponent (int8; BONS v2) - it was a
        # fixed -13 that clipped the 1.7B's gammas above 4 (layout.norm16_exp).
        b = img.by_name[w16_name]["bias"]
        return b - 256 if b >= 128 else b

    def rmsnorm_sp(m, e, name):
        n = len(m)
        if modes.get("rmsnorm", "cpu") == "hw":
            w16_name = name.replace(".weight", "") + ".w16"
            _, w16 = img.tensor(w16_name)
            slot = max(8192, ((2*n + 2047)//2048)*2048)
            sp = bytearray(3*slot)
            sp[0:2*n]         = np.asarray(m).astype("<i2").tobytes()
            sp[slot:slot+2*n] = w16.reshape(-1).astype("<i2").tobytes()
            eps_int = max(1, int(round(float(h["rms_eps"])*2.0**(-2*e))))
            words = vasm.assemble(vasm.PROGRAMS["rmsnorm"]())
            vasm.VecInterp(sp).run(words, [0, slot, 2*slot, n//16,
                                           -int(np.log2(n)), eps_int,
                                           w16_exp(w16_name) - E_NORM - 15])
            return np.frombuffer(bytes(sp[2*slot:2*slot+2*n]), dtype="<i2").astype(np.int64)
        w  = norm_w(name)
        v  = m.astype(f32)*f32(np.exp2(e))
        ss = fsum32(v*v)
        r  = f32(1.0)/np.sqrt(f32(f32(ss/f32(len(v))) + f32(h["rms_eps"])))
        o  = v*r*w*f32(np.exp2(-E_NORM))
        return np.clip(lroundf(o), -32767, 32767)

    def quant8_fixed(m):
        # Per-head attention quantization keeps the fixed >>8: the sequencer patches the
        # softmax scale (which folds in this exponent) into a program it runs head after head
        # with no CPU in the loop, so the shift has to be known before the data exists.
        # Mirrors PROGRAMS["quant8f"] / vec_quant8_head().
        return np.clip(bx.rne_shift(m, 8), -127, 127).astype(np.int8)

    def quant8_sp(m):
        # Scale to the int8 range instead of shifting by a fixed 8. After rmsnorm the mantissas
        # sit near 2**11, so >>8 handed the matvec int8 values of about +-9 - 3 of the 7 bits
        # available - and cost ~30% rel-rms on every matvec output, compounding layer over layer
        # into garbage tokens.
        #
        # Same semantics as PROGRAMS["quant8"]: pre-shift left by VCLZN(absmax) = 15 - bitlen,
        # then a fixed RNE at bit 8. sh may be negative (a small vector scales up), and the
        # absmax == 0 case takes VCLZN(0) == 0.
        amax = int(np.abs(m).max()) if len(m) else 0
        sh   = 8 if amax == 0 else amax.bit_length() - 7
        m64  = m.astype(np.int64)
        v    = (m64 << (8 - sh)) if sh <= 8 else (m64 >> (sh - 8))
        return np.clip(bx.rne_shift(v, 8), -127, 127).astype(np.int8), sh

    def norm_i32_sp(res):
        amax = int(np.abs(res).max()) if len(res) else 0
        k = 0
        while (amax >> k) > 16383:
            k += 1
        return np.clip(res.astype(np.int64) >> k, -32767, 32767), k

    def silu_sp(gm, eg, um, eu):
        if modes.get("silu_mul", "cpu") == "hw":
            n  = len(gm)
            # Regions sized to the vector: the 8 KB slots were fine for the tiny model
            # (ffn 512) and silently overlapped on the 1.7B (ffn 6144: the up vector
            # overwrote the last third of the gate vector, so the mirror's silu was wrong
            # for elements 4096.. of every layer).
            slot = ((2*n + 2047)//2048)*2048
            sp = bytearray(3*slot)
            sp[0:2*n]         = gm.astype("<i2").tobytes()
            sp[slot:slot+2*n] = um.astype("<i2").tobytes()
            words = vasm.assemble(vasm.PROGRAMS["silu_mul"]())
            vasm.VecInterp(sp).run(words, [0, slot, 2*slot, n//16, eg + 6, 0, -15])
            return np.frombuffer(bytes(sp[2*slot:2*slot+2*n]), dtype="<i2").astype(np.int64)
        g = gm.astype(f32)*f32(np.exp2(eg))
        u = um.astype(f32)*f32(np.exp2(eu))
        o = (g/(f32(1.0) + np.exp(-g, dtype=f32))*u).astype(f32)*f32(np.exp2(-(eg + eu + 15)))
        return np.clip(lroundf(o), -32767, 32767)

    def resid_sp(am, ea, bm, eb):
        # Pick the output exponent from the operands' real magnitudes, not from the worst case
        # both-mantissas-full bound. `max(ea, eb) + 1` spends a mantissa bit on every call, and
        # there are two calls per layer against a 15-bit mantissa: by layer 11 the residual was
        # down to a single bit (every element 0 or +-2**ex) and its own quantization error then
        # grew 4x per layer, which is what turned the output into garbage tokens.
        emin  = min(ea, eb)
        bound = ((int(np.abs(am).max()) << (ea - emin)) +
                 (int(np.abs(bm).max()) << (eb - emin)))
        eo    = emin + max(0, bound.bit_length() - 15)
        sa, sb = eo - ea, eo - eb
        a64 = am.astype(np.int64)
        b64 = bm.astype(np.int64)
        v   = ((a64 >> sa) if sa >= 0 else (a64 << -sa)) + \
              ((b64 >> sb) if sb >= 0 else (b64 << -sb))
        return np.clip(v, -32767, 32767), eo

    def matvec_i8(name, x8, e8):
        e = entry(name)
        scales, codes = tensor(name)
        res = bx.matvec_ternary(scales, codes, x8, e["bias"], e["rshift"])
        return res, e8 - e["bias"] + e["rshift"]

    def quantize_f_pow2(x):
        amax = np.abs(x).max()
        if amax == 0:
            return np.zeros(len(x), dtype=np.int8), 0
        e8 = int(np.frexp(f32(amax))[1]) - 7
        return np.clip(lroundf(x*f32(np.exp2(-e8))), -127, 127).astype(np.int8), e8

    n_layer         = h["n_layer"]
    n_head, n_kv    = h["n_head"], h["n_kv_head"]
    hd              = h["head_dim"]
    eps             = h["rms_eps"]
    emb_s, emb_c    = tensor("token_embd.weight")

    kc, vc, ks, vs = [], [], [], []
    for _ in range(n_layer):
        kc.append([]); vc.append([]); ks.append([]); vs.append([])
    hbm_kv = {}  # (l, kvh, kind) -> list of (ek, int8 codes) in position order.

    def kv_hbm_append(l, kvh, kind, vec):
        amax = np.abs(vec).max()
        ek = (int(np.frexp(f32(amax))[1]) - 7) if amax > 0 else 0
        codes = np.clip(lroundf(vec*f32(np.exp2(-ek))), -127, 127).astype(np.int8)
        hbm_kv.setdefault((l, kvh, kind), []).append((ek, codes))

    def kv_hbm_wv(l, kvh, pr):
        pmax = pr.max()
        ep = (int(np.frexp(f32(pmax))[1]) - 7) if pmax > 0 else 0
        p8 = np.clip(lroundf(pr*f32(np.exp2(-ep))), 0, 127).astype(np.int64)
        rowsd = hbm_kv[(l, kvh, 1)]
        ekmax = max(e for e, _ in rowsd)
        B     = 22 - ekmax
        Rpre = (((1 << -ep) + len(pr))).bit_length() + 5
        nch = h["n_channels"]
        parts = []
        for c in range(nch):
            pos = list(range(c, len(pr), nch))
            if pos:
                sc8 = np.array([[2.0**rowsd[t][0]]*(hd//32) for t in pos], dtype=np.float16)
                cd8 = np.stack([rowsd[t][1] for t in pos])
                parts.append(bx.axpy_rows(sc8, cd8, p8[pos], B, Rpre).astype(f32))
            else:
                parts.append(np.zeros(hd, dtype=f32))
        acc = np.zeros(hd, dtype=f32)
        for c in range(nch):
            acc = (acc + parts[c]).astype(f32)
        return acc*f32(np.exp2(ep + Rpre - B))

    def vec_prog(name, sp, args):
        words = vasm.assemble(vasm.PROGRAMS[name]())
        return vasm.VecInterp(sp).run(words, args)

    def res_rmsnorm_head(m, es, w16, ew16):
        n  = len(m)
        sp = bytearray(3*4096)
        sp[0:2*n]         = np.asarray(m).astype("<i2").tobytes()
        sp[4096:4096+2*n] = np.asarray(w16).astype("<i2").tobytes()
        eps_int = max(1, int(round(float(h["rms_eps"])*2.0**(-2*es))))
        vec_prog("rmsnorm", sp, [0, 4096, 8192, n//16, -int(np.log2(n)), eps_int,
                                 ew16 - E_NORM - 15])
        return np.frombuffer(bytes(sp[8192:8192+2*n]), dtype="<i2").astype(np.int64)

    def res_rope_head(m, tab_row):
        n  = len(m)
        sp = bytearray(3*4096)
        sp[0:2*n]         = np.asarray(m).astype("<i2").tobytes()
        sp[4096:4096+2*n] = np.asarray(tab_row).astype("<i2").tobytes()
        vec_prog("rope", sp, [0, n, 4096, 4096 + n, (n//2)//16, 0, n])
        return np.frombuffer(bytes(sp[0:2*n]), dtype="<i2").astype(np.int64)

    def kv_hbm_fused_int(l, kvh, q8, eq):
        # Returns (red int64, P8, Rpre, Bv): the integer pieces before normalization.
        rowsk = hbm_kv[(l, kvh, 0)]
        ekm = max(e for e, _ in rowsk)
        Bk = 22 - ekm
        boundk = (hd//32)*32*127*127*1024 << 12
        Rk = 0
        while (boundk >> Rk) >= (1 << 31):
            Rk += 1
        sck = np.array([[2.0**e]*(hd//32) for e, _ in rowsk], dtype=np.float16)
        cdk = np.stack([c for _, c in rowsk])
        res = bx.matvec_q8(sck, cdk, q8, Bk, Rk).astype(np.int64)
        cnt = len(res)
        padn = (16 - cnt % 16) % 16
        buf = np.concatenate([res, np.full(padn, -2**31, dtype=np.int64)])
        pslot = max(4096, ((4*len(buf) + 2047)//2048)*2048)   # scores, then p8 (any context)
        spb = bytearray(2*pslot)
        spb[0:4*len(buf)] = buf.astype("<i4").tobytes()
        lg = int(np.log2(hd))
        Ceff  = 523 if (lg & 1) else 739
        sigma = eq - Bk + Rk - (lg >> 1)
        s1v = 11
        words = vasm.assemble(vasm.PROGRAMS["softmax8"]())
        S, _ = vasm.VecInterp(spb).run(words, [0, Ceff, pslot, len(buf)//16,
                                               -s1v, sigma + 6 + s1v, 6144])
        p8 = np.frombuffer(bytes(spb[pslot:pslot+len(buf)]), dtype=np.int8).astype(np.int64)
        P8 = int(S[6]) if S[6] else 1
        rowsv = hbm_kv[(l, kvh, 1)]
        ekv = max(e for e, _ in rowsv)
        Bv = 22 - ekv
        Rpre = P8.bit_length() + 5
        nch = h["n_channels"]
        red = np.zeros(hd, dtype=np.int64)
        for c in range(nch):
            posl = list(range(c, cnt, nch))
            if posl:
                scv = np.array([[2.0**rowsv[t][0]]*(hd//32) for t in posl], dtype=np.float16)
                cdv = np.stack([rowsv[t][1] for t in posl])
                red += bx.axpy_rows(scv, cdv, p8[posl], Bv, Rpre).astype(np.int64)
        return red, P8, Rpre, Bv

    def kv_hbm_fused(l, kvh, qh, q8=None, eq=None):
        # Firmware's fused path: engine scores -> softmax8 (interpreter) -> AXPY -> reduce.
        if q8 is None:
            eq = 0
            amax = np.abs(qh).max()
            if amax > 0:
                eq = int(np.frexp(f32(amax))[1]) - 7
            q8 = np.clip(lroundf(qh*f32(np.exp2(-eq))), -127, 127).astype(np.int8)
        rowsk = hbm_kv[(l, kvh, 0)]
        ekm   = max(e for e, _ in rowsk)
        Bk    = 22 - ekm
        boundk = (hd//32)*32*127*127*1024 << 12
        Rk = 0
        while (boundk >> Rk) >= (1 << 31):
            Rk += 1
        sck = np.array([[2.0**e]*(hd//32) for e, _ in rowsk], dtype=np.float16)
        cdk = np.stack([c for _, c in rowsk])
        res = bx.matvec_q8(sck, cdk, q8, Bk, Rk).astype(np.int64)
        cnt  = len(res)
        padn = (16 - cnt % 16) % 16
        buf  = np.concatenate([res, np.full(padn, -2**31, dtype=np.int64)])
        pslot = max(4096, ((4*len(buf) + 2047)//2048)*2048)
        spb  = bytearray(2*pslot)
        spb[0:4*len(buf)] = buf.astype("<i4").tobytes()
        lg = int(np.log2(hd))
        Ceff  = 523 if (lg & 1) else 739
        sigma = eq - Bk + Rk - (lg >> 1)
        s1v   = 11
        words = vasm.assemble(vasm.PROGRAMS["softmax8"]())
        S, _  = vasm.VecInterp(spb).run(words, [0, Ceff, pslot, len(buf)//16,
                                                -s1v, sigma + 6 + s1v, 6144])
        p8 = np.frombuffer(bytes(spb[pslot:pslot+len(buf)]), dtype=np.int8).astype(np.int64)
        P8 = int(S[6]) if S[6] else 1
        rowsv = hbm_kv[(l, kvh, 1)]
        ekv   = max(e for e, _ in rowsv)
        Bv    = 22 - ekv
        Rpre = P8.bit_length() + 5
        nch = h["n_channels"]
        red = np.zeros(hd, dtype=np.int64)
        for c in range(nch):
            posl = list(range(c, cnt, nch))
            if posl:
                scv = np.array([[2.0**rowsv[t][0]]*(hd//32) for t in posl], dtype=np.float16)
                cdv = np.stack([rowsv[t][1] for t in posl])
                red += bx.axpy_rows(scv, cdv, p8[posl], Bv, Rpre).astype(np.int64)
        assert np.abs(red).max() < 2**31
        fs = f32(np.ldexp(f32(1.0)/f32(P8), Rpre - Bv))
        return red.astype(f32)*fs

    def rmsnorm_f(x, w):
        ss = fsum32(x*x)
        r  = f32(1.0)/np.sqrt(f32(f32(ss/f32(len(x))) + f32(eps)))
        return (x*r*w).astype(f32)

    def embed_bfp_int(tok):
        # Firmware's integer embed: (q-1)*m11 aligned to the smallest group e16, then
        # normalized to 15-bit mantissas; all integer.
        m11, e16 = bx.fp16_decompose(emb_s[tok])
        e16min = int(e16.min())
        sh  = np.minimum(e16 - e16min, 20).astype(np.int64)
        q   = emb_c[tok].astype(np.int64).reshape(-1, 128) - 1
        amax = 0
        for g in range(q.shape[0]):
            if np.any(q[g] != 0):
                gm = int(abs(int(m11[g]))) << int(sh[g])
                amax = max(amax, gm)
        bl = amax.bit_length()
        k2 = bl - 15
        out = np.zeros(q.size, dtype=np.int64)
        for g in range(q.shape[0]):
            net = int(sh[g]) - k2
            v2 = q[g]*int(m11[g])
            out[g*128:(g+1)*128] = (v2 << net) if net >= 0 else (v2 >> -net)
        return out.astype(np.int16).astype(np.int64), e16min + k2

    def forward(tok, pos, want):
        if modes.get("kv_hbm", "cpu") == "hw":
            xm, ex = embed_bfp_int(tok)
        else:
            xf = ((emb_c[tok].astype(np.int32) - 1)
                  .reshape(-1, 128)*emb_s[tok].astype(f32)[:, None]).reshape(-1).astype(f32)
            xm, ex = to_bfp16(xf)
        if trace is not None:
            trace(pos, (-1, "emb"), (xm.copy(), ex))
        for l in range(n_layer):
            p = f"blk.{l}."
            # Attention block.
            hm = rmsnorm_sp(xm, ex, p + "attn_norm.weight")
            x8, sh8 = quant8_sp(hm)
            e8 = E_NORM + sh8
            if trace is not None:
                trace(pos, (l, "attn_in_x8"), (x8.copy(), e8))
            def qkv(name):
                res, er = matvec_i8(p + name, x8, e8)
                return (res.astype(np.float64)*np.exp2(er)).astype(f32)
            if modes.get("kv_hbm", "cpu") == "hw":
                # Resident front: matvec -> norm_i32 -> int16; qk-norm/rope on the
                # interpreter (image w16 rows + rope_tab row); K/V packed from residents;
                # per-head q8 at fixed E_NORM+8; fused scores/softmax/WV as before.
                def qkv_res(name):
                    res2, _ = matvec_i8(p + name, x8, e8)
                    m2, kx = norm_i32_sp(res2)
                    e2 = entry(p + name)
                    return m2, e8 - e2["bias"] + e2["rshift"] + kx
                qm, eqr = qkv_res("attn_q.weight")
                km, ekr = qkv_res("attn_k.weight")
                vm, evr = qkv_res("attn_v.weight")
                _, qn16 = img.tensor(p + "attn_q_norm.w16")
                _, kn16 = img.tensor(p + "attn_k_norm.w16")
                eqn16, ekn16 = w16_exp(p + "attn_q_norm.w16"), w16_exp(p + "attn_k_norm.w16")
                _, rtab = img.tensor("rope_tab")
                trow = rtab[pos]
                qh8 = []
                _q16, _k16 = [], []
                for i2 in range(n_head):
                    mh = res_rmsnorm_head(qm[i2*hd:(i2+1)*hd], eqr, qn16.reshape(-1), eqn16)
                    mh = res_rope_head(mh, trow)
                    _q16.append(np.asarray(mh))
                    qh8.append((quant8_fixed(mh), E_NORM + 8))
                for i2 in range(n_kv):
                    mh = res_rmsnorm_head(km[i2*hd:(i2+1)*hd], ekr, kn16.reshape(-1), ekn16)
                    mh = res_rope_head(mh, trow)
                    _k16.append(np.asarray(mh))
                    hbm_kv.setdefault((l, i2, 0), []).append((E_NORM + 8, quant8_fixed(mh)))
                    hbm_kv.setdefault((l, i2, 1), []).append(
                        (evr + 8, quant8_fixed(vm[i2*hd:(i2+1)*hd])))
                if trace is not None:
                    trace(pos, (l, "q16"), (np.concatenate(_q16), E_NORM))
                    trace(pos, (l, "k16"), (np.concatenate(_k16), E_NORM))
                    trace(pos, (l, "v16"), (np.asarray(vm), evr))
                n_rep2 = n_head//n_kv
                rms, cs = [], []
                hmax = f32(0.0)
                for i2 in range(n_head):
                    red, P8, Rpre, Bv = kv_hbm_fused_int(l, i2//n_rep2, *qh8[i2])
                    am = int(np.abs(red).max())
                    k2 = 0
                    while (am >> k2) > 16383:
                        k2 += 1
                    rm = np.clip(red >> k2, -32767, 32767)
                    c2 = f32(np.ldexp(f32(1.0)/f32(P8), Rpre - Bv + k2))
                    hm2 = f32(f32(am >> k2)*c2)
                    if hm2 > hmax:
                        hmax = hm2
                    rms.append(rm)
                    cs.append(c2)
                ea8 = (int(np.frexp(hmax)[1]) - 7) if hmax > 0 else 0
                a8 = np.zeros(n_head*hd, dtype=np.int8)
                for i2 in range(n_head):
                    Mh = int(np.clip(lroundf(np.ldexp(cs[i2], 15 - ea8)), 0, 32767))
                    a8[i2*hd:(i2+1)*hd] = np.clip(
                        bx.rne_shift(rms[i2]*Mh, 15), -127, 127).astype(np.int8)
                # Raw (mantissa, exponent) trace points: the firmware's `dbg` checksums the
                # same scratchpad regions at the same points (software/lab/mirror.py).
                if trace is not None:
                    trace(pos, (l, "attn_x8"), (a8.copy(), ea8))
                res, er = matvec_i8(p + "attn_output.weight", a8, ea8)
                dm, kk  = norm_i32_sp(res)
                xm, ex  = resid_sp(xm, ex, dm, er + kk)
                if trace is not None:
                    trace(pos, (l, "attn_res"), (xm.copy(), ex))
                hm = rmsnorm_sp(xm, ex, p + "ffn_norm.weight")
                x8, sh8 = quant8_sp(hm)
                if trace is not None:
                    trace(pos, (l, "ffn_x8"), (x8.copy(), E_NORM + sh8))
                gres, ger = matvec_i8(p + "ffn_gate.weight", x8, E_NORM + sh8)
                gm, kg = norm_i32_sp(gres)
                ures, uer = matvec_i8(p + "ffn_up.weight", x8, E_NORM + sh8)
                um, ku = norm_i32_sp(ures)
                sm = silu_sp(gm, ger + kg, um, uer + ku)
                s8, shs = quant8_sp(sm)
                if trace is not None:
                    trace(pos, (l, "ffn_g16"), (sm.copy(), ger + kg))
                    trace(pos, (l, "silu_x8"), (s8.copy(), (ger + kg) + (uer + ku) + 15 + shs))
                dres, der = matvec_i8(p + "ffn_down.weight", s8,
                                      (ger + kg) + (uer + ku) + 15 + shs)
                dm, kd = norm_i32_sp(dres)
                xm, ex = resid_sp(xm, ex, dm, der + kd)
                if trace is not None:
                    trace(pos, (l, "ffn_res"), (xm.copy(), ex))
                continue
            if trace is not None:
                trace(pos, (l, "h_attn"), hm.astype(np.float64)*2.0**E_NORM)
            q = qkv("attn_q.weight")
            k = qkv("attn_k.weight")
            v = qkv("attn_v.weight")
            if trace is not None:
                trace(pos, (l, "qkv_raw"), np.concatenate([q, k, v]).astype(np.float64))
            qn, kn = norm_w(p + "attn_q_norm.weight"), norm_w(p + "attn_k_norm.weight")
            q = np.concatenate([rmsnorm_f(q[i*hd:(i+1)*hd], qn) for i in range(n_head)])
            k = np.concatenate([rmsnorm_f(k[i*hd:(i+1)*hd], kn) for i in range(n_kv)])
            # Rope from the table baked into the image, not recomputed here. Computing
            # theta = pos*theta_base**(-2p/hd) inline is plain RoPE: it silently drops YaRN,
            # which the 1.7B uses (factor 4, plus an mscale of ~1.1387) and the tiny sim model
            # does not (factor 1), so the tiny gate passed while every real model was wrong.
            # The firmware derives the same thing; the table is the shared source of truth.
            _, rope_tab = img.tensor("rope_tab")
            rt = np.asarray(rope_tab, dtype=np.int64).reshape(-1, hd)[pos]
            rc = (rt[:hd//2].astype(f32))*f32(2.0**-14)
            rs = (rt[hd//2:].astype(f32))*f32(2.0**-14)
            for arr, nh in ((q, n_head), (k, n_kv)):
                for i in range(nh):
                    vh = arr[i*hd:(i+1)*hd]
                    x0 = vh[:hd//2].copy()
                    x1 = vh[hd//2:].copy()
                    vh[:hd//2] = x0*rc - x1*rs
                    vh[hd//2:] = x0*rs + x1*rc
            if trace is not None:
                trace(pos, (l, "q_roped"), q.astype(np.float64))
                trace(pos, (l, "k_roped"), k.astype(np.float64))
            # Per-head scales, not one per token: the head spread of K is large (16x at layer 0
            # of the 1.7B, where head 4 peaks at 311 and head 6 at 18.9), so a single tensor
            # scale leaves the quiet heads under 3 bits and wrecks their scores.
            kr = k.reshape(n_kv, hd)
            vr = v.reshape(n_kv, hd)
            ka = np.abs(kr).max(axis=1)
            va = np.abs(vr).max(axis=1)
            kscale = np.where(ka > 0, ka/f32(127.0), f32(1.0)).astype(f32)
            vscale = np.where(va > 0, va/f32(127.0), f32(1.0)).astype(f32)
            kc[l].append(lroundf(kr*(f32(1.0)/kscale)[:, None]).astype(np.int8).reshape(-1))
            ks[l].append(kscale)
            vc[l].append(lroundf(vr*(f32(1.0)/vscale)[:, None]).astype(np.int8).reshape(-1))
            vs[l].append(vscale)
            if modes.get("kv_hbm", "cpu") == "hw":
                for i2 in range(n_kv):
                    kv_hbm_append(l, i2, 0, k[i2*hd:(i2+1)*hd])
                    kv_hbm_append(l, i2, 1, v[i2*hd:(i2+1)*hd])
            n_rep = n_head//n_kv
            attn  = np.zeros(n_head*hd, dtype=f32)
            for i in range(n_head):
                kvh = i//n_rep
                qh  = q[i*hd:(i+1)*hd]
                if modes.get("kv_hbm", "cpu") == "hw":
                    attn[i*hd:(i+1)*hd] = kv_hbm_fused(l, kvh, qh)
                    continue
                sc  = np.array([f32(f32(fsum32(qh*kt[kvh*hd:(kvh+1)*hd]))*kst[kvh])
                                for kt, kst in zip(kc[l], ks[l])], dtype=f32)
                sc  = sc*f32(1.0/np.sqrt(hd))
                e   = np.exp(sc - sc.max(), dtype=f32)
                pr  = e*f32(f32(1.0)/fsum32(e))
                if modes.get("kv_hbm", "cpu") == "hw":
                    attn[i*hd:(i+1)*hd] = kv_hbm_wv(l, kvh, pr)
                else:
                    ah  = np.zeros(hd, dtype=f32)
                    for t in range(len(vc[l])):
                        ah += pr[t]*vs[l][t][kvh]*vc[l][t][kvh*hd:(kvh+1)*hd].astype(f32)
                    attn[i*hd:(i+1)*hd] = ah
            if trace is not None:
                trace(pos, (l, "attn_out"), attn.astype(np.float64))
            a8, ea8 = quantize_f_pow2(attn)
            res, er = matvec_i8(p + "attn_output.weight", a8, ea8)
            dm, kk  = norm_i32_sp(res)
            xm, ex  = resid_sp(xm, ex, dm, er + kk)
            if trace is not None:
                trace(pos, (l, "resid_attn"), xm.astype(np.float64)*2.0**ex)
            # FFN block (fully resident).
            hm = rmsnorm_sp(xm, ex, p + "ffn_norm.weight")
            x8, sh8 = quant8_sp(hm)
            gres, ger = matvec_i8(p + "ffn_gate.weight", x8, E_NORM + sh8)
            gm, kg = norm_i32_sp(gres)
            ures, uer = matvec_i8(p + "ffn_up.weight", x8, E_NORM + sh8)
            um, ku = norm_i32_sp(ures)
            sm = silu_sp(gm, ger + kg, um, uer + ku)
            es = (ger + kg) + (uer + ku) + 15
            s8, shs = quant8_sp(sm)
            dres, der = matvec_i8(p + "ffn_down.weight", s8, es + shs)
            dm, kd = norm_i32_sp(dres)
            xm, ex = resid_sp(xm, ex, dm, der + kd)
            if trace is not None:
                trace(pos, (l, "ffn"), xm.astype(np.float64)*2.0**ex)
        if not want:
            return -1
        hm = rmsnorm_sp(xm, ex, "output_norm.weight")
        x8, _ = quant8_sp(hm)
        head = "output.weight" if "output.weight" in img.by_name else "token_embd.weight"
        e = entry(head)
        scales, codes = tensor(head)
        res = bx.matvec_ternary(scales, codes, x8, e["bias"], e["rshift"])
        return int(np.argmax(res[:h["vocab"]]))

    pos = 0
    tok = -1
    for i, t in enumerate(prompt):
        tok = forward(t, pos, i == len(prompt) - 1)
        pos += 1
    out = []
    for _ in range(n_new):
        out.append(tok)
        tok = forward(tok, pos, True)
        pos += 1
    return out

# Hybrid Mirror ------------------------------------------------------------------------------------

def _fsum32(v):
    acc = f32(0.0)
    for e in np.asarray(v, dtype=f32):
        acc = f32(acc + e)
    return acc


def _sumsq64(v):
    """The canonical sum of squares (doc/units.md (Canonical reductions), hybrid.c sumsq64)."""
    v = np.asarray(v, dtype=f32)
    sq = (v*v).astype(f32)
    P = np.zeros(64, dtype=f32)
    full = len(sq)//64*64
    for row in sq[:full].reshape(-1, 64):
        P = (P + row).astype(f32)
    tail = len(sq) - full
    if tail:
        P[:tail] = (P[:tail] + sq[full:]).astype(f32)
    w = 32
    while w:
        P[:w] = (P[:w] + P[w:2*w]).astype(f32)
        w //= 2
    return P[0]


def _lroundf(x):
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int64)


def hybrid27_generate(img, prompt, n_new, exact=False, fast=False, fast_stats=None, fast_cfg=None,
                      trace=None, silu_mode=None):
    """exact=True: fp32 weights and no activation quantization (the golden's arithmetic),
    which tests the mirror's structure against the golden independently of the int8 noise.
    fast=True: the DeltaNet recurrence in the fast path's arithmetic (doc/units.md (DeltaNet recurrence arithmetic)) -
    q and k as int8 per head (unit vectors, exponent -7), v as int8 per head, the state as
    int16 with one exponent per head that ratchets up when a row would overflow, the decay
    as Q15, the dots as exact int32 sums (the engine's Q8 matvec on the two int8 planes),
    the rank-1 update in the state's fixed point (the vector unit). Everything else stays
    as in the CPU path. fast_stats (a dict) collects the state exponents seen. fast_cfg
    picks the precisions: qk_bits (8 = one int8 plane, 16 = two), wq_extra (extra fraction
    bits on the rank-1 weight, applied as a shift after the product), state_bits (the
    ratchet's limit), decay_q (the decay's fixed point, None = fp32)."""
    silu_mode = silu_mode or ("mathematical" if exact else "rv32f")
    if silu_mode not in ("mathematical", "rv32f"):
        raise ValueError("unknown SiLU numerical profile: " + silu_mode)
    h = img.header
    assert h["version"] >= 3 and (any(t == "deltanet" for t in h["layer_types"]) or h["hadamard_block"])
    n_layer, n_embd = h["n_layer"], h["n_embd"]
    n_head, n_kv, hd = h["n_head"], h["n_kv_head"], h["head_dim"]
    nk, hk, nv, hv = h["dn_num_k_heads"], h["dn_k_head_dim"], h["dn_num_v_heads"], h["dn_v_head_dim"]
    nq, nvh, K = nk*hk, nv*hv, h["dn_conv_kernel"]
    rope_n = h["rope_dims"] or hd
    eps = f32(h["rms_eps"])
    B = h["hadamard_block"]
    widths = list(h["hadamard_widths"])
    gated = bool(h["attn_gated"])
    n_rep = n_head//n_kv

    def entry(role, l, kind=None):
        for e in img.entries:
            if e["role"] == role and e["layer"] == l and (kind is None or e["kind"] == kind):
                return e
        return None

    def f32_rows(role, l):
        e = entry(role, l, layout.KIND_F32)
        assert e is not None, (role, l)
        _, v = img.tensor(e["name"])
        return np.asarray(v, dtype=f32)

    def quantize_f_pow2(x):
        amax = np.abs(x).max()
        if amax == 0:
            return np.zeros(len(x), dtype=np.int8), 0
        e8 = int(np.frexp(f32(amax))[1]) - 7
        return np.clip(_lroundf(x*f32(np.exp2(-e8))), -127, 127).astype(np.int8), e8

    wcache = {}

    def ksegs(e):
        """The K-segments of a wide matvec (layout.FLAG_KSEG), in order; [e] otherwise."""
        if not (e["flags"] & layout.FLAG_KSEG):
            return [e]
        segs = [t for t in img.entries if t["role"] == e["role"] and t["layer"] == e["layer"]
                and t["kind"] == e["kind"] and (t["flags"] & layout.FLAG_KSEG)]
        return sorted(segs, key=lambda t: t["kseg"])

    def matvec_f(e, x):
        segs = ksegs(e)
        if exact:
            if e["name"] not in wcache:
                ws = []
                for t in segs:
                    scales, codes = img.tensor(t["name"])
                    g = 128 if t["kind"] == layout.KIND_TERNARY else 32
                    off = 1.0 if t["kind"] == layout.KIND_TERNARY else 0.0
                    w = (codes.astype(f32).reshape(t["rows"], -1, g) - f32(off))*scales.astype(f32)[:, :, None]
                    ws.append(w.reshape(t["rows"], t["cols"]))
                wcache[e["name"]] = np.concatenate(ws, axis=1)
            return (wcache[e["name"]] @ np.asarray(x, dtype=f32)).astype(f32)
        # As infer_matvec_f: one exponent for the whole input; per segment the engine's
        # int32 result (each segment carries the tensor's bias/rshift); the sum, scaled once.
        x8, e8 = quantize_f_pow2(np.asarray(x, dtype=f32))
        res, c0 = None, 0
        for t in segs:
            scales, codes = img.tensor(t["name"])
            xs = x8[c0:c0 + t["cols"]]
            if t["kind"] == layout.KIND_TERNARY:
                r = bx.matvec_ternary(scales, codes, xs, t["bias"], t["rshift"])
            else:
                r = bx.matvec_q8(scales, codes, xs, t["bias"], t["rshift"])
            res = r.astype(np.int64) if res is None else res + r.astype(np.int64)
            c0 += t["cols"]
        sc = f32(np.exp2(f32(e8 + e["rshift"] - e["bias"])))
        return (res.astype(np.int32).astype(f32)*sc).astype(f32)

    def rotate(x, width):
        if not B:
            return np.asarray(x, dtype=f32)
        wi = widths.index(width)
        out = np.empty(width, dtype=f32)
        for b in range(width//B):
            e = entry(layout.ROLE_HADAMARD, layout.hadamard_layer(wi, b))
            out[b*B:(b + 1)*B] = matvec_f(e, x[b*B:(b + 1)*B])
        return out

    def total_cols(e):
        return sum(t["cols"] for t in ksegs(e))

    def proj(role, l, x):
        e = entry(role, l)
        return matvec_f(e, rotate(x, total_cols(e)) if B else x)

    def rmsnorm(x, w):
        ss = _sumsq64(x)
        r = f32(1.0)/f32(np.sqrt(f32(ss/f32(len(x))) + eps))
        return (x*r*w).astype(f32)

    def l2norm(x):
        nrm = f32(np.sqrt(_sumsq64(x)))
        nrm = max(nrm, eps)
        return (x*f32(f32(1)/nrm)).astype(f32)     # canonical: reciprocal, then multiply

    def sigmoid(x): return (f32(1.0)/(f32(1.0) + np.exp(-x, dtype=f32))).astype(f32)
    if silu_mode == "rv32f":
        from model.rv32_math import sigmoid_rv32 as sigmoid   # canonical LUT sigmoid
    if silu_mode == "rv32f":
        from model.rv32_math import silu_rv32 as silu
    else:
        def silu(x): return (x/(f32(1.0) + np.exp(-x, dtype=f32))).astype(f32)
    def softplus(x): return np.where(x > 20, x, np.log1p(np.exp(x, dtype=f32), dtype=f32)).astype(f32)
    def softmax_exp(x): return np.exp(x, dtype=f32)
    if silu_mode == "rv32f":
        from model.rv32_math import softmax_exp_rv32 as softmax_exp   # canonical

    # Tables.
    norm_attn = [f32_rows(layout.ROLE_ATTN_NORM, l).reshape(-1) for l in range(n_layer)]
    norm_ffn  = [f32_rows(layout.ROLE_FFN_NORM, l).reshape(-1) for l in range(n_layer)]
    dn = [t == "deltanet" for t in h["layer_types"]]
    norm_q, norm_k, norm_ssm, ssm_a, ssm_dt, conv_w = {}, {}, {}, {}, {}, {}
    for l in range(n_layer):
        if dn[l]:
            norm_ssm[l] = f32_rows(layout.ROLE_SSM_NORM, l).reshape(-1)
            ssm_a[l]    = f32_rows(layout.ROLE_SSM_A, l).reshape(-1)
            ssm_dt[l]   = f32_rows(layout.ROLE_SSM_DT, l).reshape(-1)
            conv_w[l]   = f32_rows(layout.ROLE_SSM_CONV, l).reshape(-1)   # c*K + i
        else:
            norm_q[l] = f32_rows(layout.ROLE_ATTN_Q_NORM, l).reshape(-1)
            norm_k[l] = f32_rows(layout.ROLE_ATTN_K_NORM, l).reshape(-1)
    out_norm = f32_rows(layout.ROLE_OUT_NORM, 0).reshape(-1)
    emb16  = entry(layout.ROLE_TOK_EMB16, 0)
    emb16x = entry(layout.ROLE_TOK_EMB16_EXP, 0)
    _, mant = img.tensor(emb16["name"])
    _, exps = img.tensor(emb16x["name"])
    exps = np.asarray(exps).reshape(-1)
    rope_freq = np.array([f32(np.float32(h["rope_theta"])**f32(-2.0*i/rope_n)) for i in range(rope_n//2)], dtype=f32)

    # State.
    state = {l: np.zeros((nv, hv, hk), dtype=f32) for l in range(n_layer) if dn[l]}
    win   = {l: np.zeros((K - 1, 2*nq + nvh), dtype=f32) for l in range(n_layer) if dn[l]}
    kc    = {l: [] for l in range(n_layer) if not dn[l]}
    vc    = {l: [] for l in range(n_layer) if not dn[l]}
    scale_dn = f32(1.0/np.sqrt(hk))
    scale_at = f32(1.0/np.sqrt(hd))

    cfg = dict(qk_bits=8, wq_extra=0, state_bits=16, decay_q=15, mode="planes")
    cfg.update(fast_cfg or {})
    qk_bits, wq_extra, state_bits, decay_q = (cfg[k] for k in ("qk_bits", "wq_extra", "state_bits", "decay_q"))
    vec_qk_bits = cfg.get("vec_qk_bits", 16)
    state_lim = 1 << (state_bits - 1)
    vecunit = cfg["mode"] == "vecunit"

    def q8_head(x, bits=8):
        """int8 with the head's own power-of-two exponent (mantissas in [64, 128));
        bits=16 is the two-plane form (mantissas in [16384, 32768))."""
        amax = np.abs(x).max()
        if amax == 0:
            return np.zeros(len(x), dtype=np.int32), 0
        e = int(np.frexp(f32(amax))[1]) - (bits - 1)
        lim = (1 << (bits - 1)) - 1
        return np.clip(_lroundf(x*f32(np.exp2(-e))), -lim, lim).astype(np.int32), e

    state16 = {l: np.zeros((nv, hv, hk), dtype=np.int64) for l in range(n_layer) if dn[l]}
    state_e = {l: np.full(nv, -20, dtype=np.int64) for l in range(n_layer) if dn[l]}

    def _shr(x, n):
        """Arithmetic shift by n: right (floor) for n > 0, left for n < 0 - SSHLV/VSHLS."""
        x = np.asarray(x, dtype=np.int64)
        return x >> n if n >= 0 else x << (-n)

    def _rne(x, n):
        """Round-nearest-even right shift by n >= 0 (VRNE)."""
        x = np.asarray(x, dtype=np.int64)
        if n == 0:
            return x
        q, rem = x >> n, x & ((1 << n) - 1)
        half = 1 << (n - 1)
        return q + ((rem > half) | ((rem == half) & (q & 1 == 1)))

    def recurrence_vecunit(l, hh, qh, kk, vh, decay, beta_h):
        """One head in the two-pass vector-unit arithmetic (doc/units.md (DeltaNet recurrence arithmetic)): the state
        int16 in the scratchpad at one exponent per head, k and q as int16 with their own
        exponents, the dots exact into the 48-bit scalar registers, the per-row delta in
        units of 2^-19 (a scalar SSHLV, VMULH15 by the Q15 decay and by -beta, both
        floors), the exponent ratchet decided by the CPU between the passes from the
        head's max |state| and max |delta| (REDMAX), the update as one VMUL/VMUL/VADD/VRNE
        per beat with the decay's Q15 constant carrying the ratchet shift. Returns o."""
        k16, ek = q8_head(kk, vec_qk_bits)
        q16, eq = q8_head(qh, vec_qk_bits)
        M = state16[l][hh]
        eM = int(state_e[l][hh])
        d15 = int(np.clip(_lroundf(f32(decay)*f32(32768.0)), 0, 32767))
        bq = int(np.clip(_lroundf(f32(beta_h)*f32(32768.0)), 0, 32767))
        # Pass A: the k dots (exact), max |M|, the per-row delta.
        dot = M.astype(np.int64) @ k16.astype(np.int64)                 # units 2^(eM+ek)
        assert np.abs(dot).max() < (1 << 47), (l, hh, vec_qk_bits, np.abs(dot).max())
        maxM = int(np.abs(M).max())
        r = -19 - (eM + ek)
        dot_r = _shr(dot, r)                                            # units 2^-19
        assert np.abs(dot_r).max() < (1 << 31), (l, hh, eM, ek, np.abs(dot_r).max())
        ddr = (dot_r*d15) >> 15                                         # VMULH15 (floor)
        nv_ = -_lroundf(vh.astype(np.float64)*float(1 << 19))
        assert np.abs(nv_).max() < (1 << 31)
        s3 = ddr + nv_                                                  # decay*dot - v
        delta_int = (s3*(-bq)) >> 15                                    # (v - decay*dot)*beta
        maxD = int(np.abs(delta_int).max())
        # The CPU: the ratchet from the two maxima (a bound, not the exact new max).
        kmax = float(np.abs(k16).max())*float(np.exp2(ek))
        t = (maxM*float(decay) + maxD*float(np.exp2(-19))*kmax*float(np.exp2(-eM)))/float((1 << 15) - 4)
        n = 0 if t <= 1.0 else int(np.ceil(np.log2(t)))
        eM2 = eM + n
        d15n = int(np.clip(_lroundf(f32(decay)*f32(np.exp2(15 - n))), 0, 32767))
        sh = eM2 + 4 - ek
        # Pass B: the row update and the q dots.
        rnd = (1 << (sh - 1)) if sh > 0 else 0
        delta_s = _shr(delta_int + rnd, sh)
        assert np.abs(delta_s).max() < (1 << 16), (l, hh, np.abs(delta_s).max(), sh, n)
        prod = M.astype(np.int64)*d15n + np.outer(delta_s, k16.astype(np.int64))
        newM = _rne(prod, 15)
        assert np.abs(newM).max() <= 32767, (l, hh, np.abs(newM).max(), n, t)
        M[:] = newM
        state_e[l][hh] = eM2
        if fast_stats is not None:
            fast_stats.setdefault("eM", []).append(eM2)
            fast_stats.setdefault("n", []).append(n)
        o = M.astype(np.int64) @ q16.astype(np.int64)                   # units 2^(eM2+eq)
        # Keep the scalar result in int32.  For the hardware q16 form this is 8; a q24
        # precision probe models the same 48-bit REDSUM with two mantissa planes and uses 16.
        o_shift = vec_qk_bits - 8
        o_r = o >> o_shift
        assert np.abs(o_r).max() < (1 << 31)
        of = (o_r.astype(np.float64)*float(np.exp2(eM2 + eq + o_shift))).astype(f32)
        return (of*scale_dn).astype(f32)

    def recurrence_fast(l, hh, qh, kk, vh, decay, beta_h):
        """One head, the fast path's arithmetic. Returns o (float, hv)."""
        if vecunit:
            return recurrence_vecunit(l, hh, qh, kk, vh, decay, beta_h)
        k8, ek = q8_head(kk, qk_bits)
        q8, eq = q8_head(qh, qk_bits)
        M = state16[l][hh]
        eM = int(state_e[l][hh])
        if decay_q is None:
            M[:] = _lroundf(M.astype(np.float64)*float(decay)).astype(np.int64)
        else:
            # decay as Q15: M <- (M * d15 + 2^14) >> 15 (the vector unit's multiply-shift).
            one = 1 << decay_q
            dq = int(np.clip(_lroundf(f32(decay)*f32(one)), 0, one - 1))
            M[:] = (M.astype(np.int64)*dq + (one >> 1)) >> decay_q
        # dot_j = M_j . k  (int32 sums; units 2^(eM+ek))
        dot = (M.astype(np.int64) @ k8.astype(np.int64))
        dotf = (dot.astype(np.float64)*float(np.exp2(eM + ek))).astype(f32)
        delta = ((vh - dotf)*beta_h).astype(f32)
        # rank-1 update: M_ji += (wq_j * k8_i) >> wq_extra with wq_j = round(delta_j *
        # 2^(ek - eM + wq_extra)) - and the exponent ratchet: if a row would overflow the
        # state's width, halve the state (eM += 1) first.
        while True:
            wq = _lroundf(delta.astype(np.float64)*float(np.exp2(ek - eM + wq_extra)))
            prod = np.outer(wq.astype(np.int64), k8.astype(np.int64))
            if wq_extra:
                prod = (prod + (1 << (wq_extra - 1))) >> wq_extra
            newM = M + prod
            if np.abs(newM).max() < state_lim and np.abs(wq).max() < (1 << 30):
                break
            M[:] = (M + 1) >> 1
            eM += 1
        M[:] = newM.astype(np.int64)
        state_e[l][hh] = eM
        if fast_stats is not None:
            fast_stats.setdefault("eM", []).append(eM)
        o = (M.astype(np.int64) @ q8.astype(np.int64)).astype(np.float64)*float(np.exp2(eM + eq))
        return (o.astype(f32)*scale_dn).astype(f32)

    def deltanet(l, hn):
        C = 2*nq + nvh
        qkv = proj(layout.ROLE_ATTN_QKV, l, hn)
        z   = proj(layout.ROLE_ATTN_ZGATE, l, hn)
        beta  = sigmoid(matvec_f(entry(layout.ROLE_SSM_BETA, l), hn))
        alpha = matvec_f(entry(layout.ROLE_SSM_ALPHA, l), hn)
        g = (softplus(alpha + ssm_dt[l])*ssm_a[l]).astype(f32)
        w = conv_w[l]
        conv = np.empty(C, dtype=f32)
        for c in range(C):
            acc = f32(0.0)
            for i in range(K - 1):
                acc = f32(acc + f32(w[c*K + i]*win[l][i, c]))
            acc = f32(acc + f32(w[c*K + K - 1]*qkv[c]))
            conv[c] = acc
        conv = silu(conv)
        win[l][:-1] = win[l][1:]
        win[l][-1] = qkv
        q = conv[:nq].copy(); k = conv[nq:2*nq].copy(); v = conv[2*nq:].copy()
        for hh in range(nk):
            q[hh*hk:(hh + 1)*hk] = l2norm(q[hh*hk:(hh + 1)*hk])
            k[hh*hk:(hh + 1)*hk] = l2norm(k[hh*hk:(hh + 1)*hk])
        rep = nv//nk
        og = np.zeros(nvh, dtype=f32)
        for hh in range(nv):
            kh = hh % nk
            qh, kk, vh = q[kh*hk:(kh + 1)*hk], k[kh*hk:(kh + 1)*hk], v[hh*hv:(hh + 1)*hv]
            decay = f32(softmax_exp(np.array([g[hh]], dtype=f32))[0])   # canonical exp (<= 0)
            if fast:
                oh = recurrence_fast(l, hh, qh, kk, vh, decay, beta[hh])
            else:
                M = state[l][hh]
                delta = np.empty(hv, dtype=f32)
                for j in range(hv):
                    Mj = M[j]
                    dot = f32(0.0)
                    for i in range(hk):
                        Mj[i] = f32(Mj[i]*decay)
                        dot = f32(dot + f32(Mj[i]*kk[i]))
                    delta[j] = f32(f32(vh[j] - dot)*beta[hh])
                oh = np.empty(hv, dtype=f32)
                for j in range(hv):
                    Mj = M[j]
                    dot = f32(0.0)
                    for i in range(hk):
                        Mj[i] = f32(Mj[i] + f32(delta[j]*kk[i]))
                        dot = f32(dot + f32(Mj[i]*qh[i]))
                    oh[j] = f32(dot*scale_dn)
            nrm = rmsnorm(oh, norm_ssm[l])
            gpos = kh*rep + hh//nk
            og[gpos*hv:(gpos + 1)*hv] = (nrm*silu(z[hh*hv:(hh + 1)*hv])).astype(f32)
        return proj(layout.ROLE_SSM_OUT, l, og)

    def attention(l, hn, pos):
        qf = proj(layout.ROLE_ATTN_Q, l, hn)
        k  = proj(layout.ROLE_ATTN_K, l, hn)
        v  = proj(layout.ROLE_ATTN_V, l, hn)
        if gated:
            qf = qf.reshape(n_head, 2*hd)
            q, gate = qf[:, :hd].copy(), qf[:, hd:].copy()
        else:
            q, gate = qf.reshape(n_head, hd).copy(), None
        k = k.reshape(n_kv, hd).copy(); v = v.reshape(n_kv, hd).copy()
        cos = np.cos((f32(pos)*rope_freq).astype(f32), dtype=f32)
        sin = np.sin((f32(pos)*rope_freq).astype(f32), dtype=f32)
        half = rope_n//2
        def rope(vh):
            x0, x1 = vh[:half].copy(), vh[half:rope_n].copy()
            vh[:half]       = (x0*cos - x1*sin).astype(f32)
            vh[half:rope_n] = (x0*sin + x1*cos).astype(f32)
        for hh in range(n_head):
            q[hh] = rmsnorm(q[hh], norm_q[l]); rope(q[hh])
        for hh in range(n_kv):
            k[hh] = rmsnorm(k[hh], norm_k[l]); rope(k[hh])
        kc[l].append(k); vc[l].append(v)
        T = len(kc[l])
        out = np.zeros((n_head, hd), dtype=f32)
        for hh in range(n_head):
            kvh = hh//n_rep
            sc = np.array([f32(_fsum32(q[hh]*kc[l][t][kvh])*scale_at) for t in range(T)], dtype=f32)
            mx = sc.max()
            ex = softmax_exp((sc - mx).astype(f32))
            s = _fsum32(ex)
            o = np.zeros(hd, dtype=f32)
            for t in range(T):
                p = f32(ex[t]/s)
                o = (o + p*vc[l][t][kvh]).astype(f32)
            if gated:
                o = (o*sigmoid(gate[hh])).astype(f32)
            out[hh] = o
        return proj(layout.ROLE_ATTN_OUT, l, out.reshape(-1))

    def checkpoint(pos, layer, tag, values, dtype="f32"):
        if trace is not None:
            trace(pos, layer, tag, dtype, np.asarray(values).copy())

    def forward(tok, pos, want_logits=True):
        x = (mant[tok].astype(f32)*f32(np.exp2(f32(exps[tok])))).astype(f32)
        checkpoint(pos, -1, "emb", x)
        for l in range(n_layer):
            hn = rmsnorm(x, norm_attn[l])
            checkpoint(pos, l, "attn_norm", hn)
            y = deltanet(l, hn) if dn[l] else attention(l, hn, pos)
            checkpoint(pos, l, "dn_out" if dn[l] else "attn_out", y)
            x = (x + y).astype(f32)
            checkpoint(pos, l, "attn_res", x)
            hn = rmsnorm(x, norm_ffn[l])
            checkpoint(pos, l, "ffn_norm", hn)
            r = rotate(hn, n_embd) if B else hn
            gate = matvec_f(entry(layout.ROLE_FFN_GATE, l), r)
            up   = matvec_f(entry(layout.ROLE_FFN_UP, l), r)
            checkpoint(pos, l, "ffn_gate", gate)
            checkpoint(pos, l, "ffn_up", up)
            act  = (silu(gate)*up).astype(f32)
            checkpoint(pos, l, "ffn_act", act)
            y = proj(layout.ROLE_FFN_DOWN, l, act)
            checkpoint(pos, l, "ffn_down", y)
            x = (x + y).astype(f32)
            checkpoint(pos, l, "ffn_res", x)
        if not want_logits:
            return None
        hn = rmsnorm(x, out_norm)
        checkpoint(pos, -1, "out_norm", hn)
        logits = proj(layout.ROLE_LM_HEAD if entry(layout.ROLE_LM_HEAD, 0) else layout.ROLE_TOK_EMB, 0, hn)
        checkpoint(pos, -1, "logits", logits[:h["vocab"]])
        best = int(np.argmax(logits[:h["vocab"]]))
        checkpoint(pos, -1, "argmax", [best], "u32")
        return best

    out, pos = [], 0
    tok = None
    for i, t in enumerate(prompt):
        tok = forward(t, pos, i == len(prompt) - 1); pos += 1
    for i in range(n_new):
        out.append(tok)
        if i + 1 < n_new:  # Match firmware: no unused forward after the final output.
            tok = forward(tok, pos); pos += 1
    return out
