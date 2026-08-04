#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# Slow end-to-end check (needs a built Verilator sim): runs the firmware in simulation and
# verifies the self-test checksums against the golden model + image bytes, and the generated
# token ids against a hybrid reference (engine-exact matvecs + float vector ops).
# Run with: .venv/bin/python -m pytest test/test_sim_smoke.py --runslow

import os
import subprocess

import numpy as np
import pytest

from trixium import layout
from trixium.model import bitexact as bx
from trixium.model.bons import BONSReader

SIM_DIR = "build/sim/gateware"
# Override to gate against a different channel count (rebuild the sim with the same
# image first: BONSAI_SIM_IMAGE=... python3 sim/bonsai_sim.py).
IMAGE   = os.environ.get("BONSAI_SIM_IMAGE", "models/tiny/bonsai-tiny-n2.img")

def fnv1a(data):
    h = 2166136261
    for b in bytes(data):
        h = ((h ^ b)*16777619) & 0xFFFFFFFF
    return h

def x_pattern(n):
    x = np.array([((i*137 + 11) & 0xFF) for i in range(n)], dtype=np.uint8)
    return x.view(np.int8)

@pytest.mark.slow
def test_sim_selftest_checksums():
    assert os.path.exists(os.path.join(SIM_DIR, "obj_dir/Vsim")), "build the sim first"
    assert os.path.exists(IMAGE)

    # Run the sim until the boot generation demo completes (the token gate needs the GEN
    # line, which lands ~50 sim-seconds / ~5 min wall after SELFTEST).
    p = subprocess.Popen(["./obj_dir/Vsim"], cwd=SIM_DIR,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    import time as _t
    deadline = _t.time() + 40*60
    try:
        for line in p.stdout:
            lines.append(line)
            if "BONSAI-READY" in line or len(lines) > 400000 or _t.time() > deadline:
                break
    finally:
        p.kill()
    out = "".join(lines)
    assert "SPTEST OK" in out, out[-2000:]
    assert "VECTEST OK" in out, "vec unit smoke failed (emission-level datapath break)"

    # Per-program CRC sweep vs the interpreter (named failure per program).
    import re as _re
    from trixium.model import asm as _vasm
    sp = bytearray(256*1024)
    seed = 0x12345678
    for i in range(1024):
        seed = (seed*1664525 + 1013904223) & 0xFFFFFFFF
        sp[0x8000 + 4*i:0x8000 + 4*i + 4] = seed.to_bytes(4, "little")
    _progs = [
        ("quant8",   "quant8",     [0x8000, 0, 0xA000, 8]),
        ("norm_i32", "norm_i32",   [0x8000, 0, 0xA200, 4, 13]),
        ("resid",    "resid_add",  [0x8000, 0x8400, 0xA400, 8, 1, 2]),
        ("silu",     "silu_mul",   [0x8000, 0x8400, 0xA600, 8, 6, 0, -15]),
        ("rmsnorm",  "rmsnorm",    [0x8000, 0x8400, 0xA800, 8, -7, 17, -17]),
        ("rope",     "rope",       [0x8000, 0x8080, 0x8400, 0x8480, 4, 0xAA00, 0xAA80]),
        ("softmax8", "softmax8",   [0x8000, 739, 0xAC00, 4, -11, 17]),
        ("reduce",   "reduce_acc", [0xA200, 0x8400, 0xAE00, 4]),
        ("scale8",   "scale8",     [0x8000, 0, 0xB000, 8, 19571]),
    ]
    exp_crc = {}
    for name, prog, pargs in _progs:
        S, _ = _vasm.VecInterp(sp).run(_vasm.assemble(_vasm.PROGRAMS[prog]()), list(pargs))
        crc = 0
        for i in range(128):
            w = int.from_bytes(sp[pargs[2] + 4*i:pargs[2] + 4*i + 4], "little")
            crc = ((crc ^ w)*16777619) & 0xFFFFFFFF
        exp_crc[name] = (crc, S[6] & 0xFFFFFFFF)
    for m2 in _re.finditer(r"VECCRC (\w+) ([0-9a-f]{8}) ([0-9a-f]{8})", out):
        name, crc, r6 = m2.group(1), int(m2.group(2), 16), int(m2.group(3), 16)
        assert name in exp_crc, name
        assert crc == exp_crc[name][0], (
            f"vec program '{name}' diverged from the interpreter "
            f"(crc {crc:08x} != {exp_crc[name][0]:08x}) - emission-level break")
    assert len(_re.findall(r"VECCRC", out)) == len(_progs), "missing VECCRC lines"

    img  = BONSReader(IMAGE)
    e    = img.by_name["blk.0.attn_q.weight"]
    n_ch = img.header["n_channels"]

    # Expected MATVEC checksum.
    scales, codes = img.tensor("blk.0.attn_q.weight")
    x8   = x_pattern(e["cols"])
    res  = bx.matvec_ternary(scales, codes, x8, e["bias"], e["rshift"])
    want = f"MATVEC CSUM={fnv1a(res.astype('<i4').tobytes()):08x}"
    assert want in out, f"{want} not found in:\n{out[-1500:]}"

    # Expected FETCH checksum: first 8 beats of channel 0's slice of the tensor.
    data = bytes(img.tensor_channel_bytes("blk.0.attn_q.weight", 0)[:8*32])
    want = f"FETCH CSUM={fnv1a(data):08x}"
    assert want in out, f"{want} not found in:\n{out[-1500:]}"

    # Generated tokens vs the hybrid reference (engine-exact matvecs + float vector ops).
    import re
    m = re.search(r"GEN:((?: \d+)+)", out)
    assert m, f"no GEN line in:\n{out[-1500:]}"
    got_tokens = [int(t) for t in m.group(1).split()]
    ref_tokens = hybrid_generate(img, [1, 2, 3], len(got_tokens))
    assert got_tokens == ref_tokens, f"sim {got_tokens} != ref {ref_tokens}"

    # HW-ops leg (present when the sim SoC has the vector unit): same greedy tokens. The
    # quantized HW ops are not bit-identical to soft-float, but on every model tested the
    # argmax agrees; a divergence here is a regression signal to investigate either way.
    mhw = re.search(r"GEN-HW:((?: \d+)+)", out)
    assert mhw, f"no GEN-HW line in:\n{out[-1500:]}"
    hw_tokens = [int(t) for t in mhw.group(1).split()]
    ref_hw    = hybrid_generate(img, [1, 2, 3], len(hw_tokens), modes={"silu_mul": "hw", "rmsnorm": "hw", "kv_hbm": "hw"})
    assert hw_tokens == ref_hw, f"HW-ops tokens {hw_tokens} != hw ref {ref_hw}"

    # Speculative-decode leg (when built): greedy speculation is lossless, so tokens
    # must equal the plain HW-ops reference at its length. SPEC counters just have to
    # show the machinery ran (acc+rej >= 0; a hit is model-dependent).
    msp = re.search(r"GEN-SP:((?: \d+)+)", out)
    if msp:
        sp_tokens = [int(t) for t in msp.group(1).split()]
        ref_sp    = hybrid_generate(img, [1, 2, 3], len(sp_tokens),
                                    modes={"silu_mul": "hw", "rmsnorm": "hw", "kv_hbm": "hw"})
        assert sp_tokens == ref_sp, f"spec tokens {sp_tokens} != ref {ref_sp}"
        assert re.search(r"SPEC acc=\d+ rej=\d+", out)

def hybrid_generate(img, prompt, n_new, modes=None, trace=None):
    """Python mirror of the firmware loop (resident-BFP): engine-bitexact matvecs,
    int16 scratchpad-resident activations with integer exponent algebra, float32 CPU op
    math mirroring the soft-float firmware, per-op cpu/hw switches (hw ops run the actual
    microcode interpreter). modes: dict op->"cpu"|"hw" (default all-cpu)."""
    from trixium.model import asm as vasm
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

    def rmsnorm_sp(m, e, name):
        n = len(m)
        if modes.get("rmsnorm", "cpu") == "hw":
            _, w16 = img.tensor(name.replace(".weight", "") + ".w16")
            sp = bytearray(3*8192)
            sp[0:2*n]         = np.asarray(m).astype("<i2").tobytes()
            sp[8192:8192+2*n] = w16.reshape(-1).astype("<i2").tobytes()
            eps_int = max(1, int(round(float(h["rms_eps"])*2.0**(-2*e))))
            words = vasm.assemble(vasm.PROGRAMS["rmsnorm"]())
            vasm.VecInterp(sp).run(words, [0, 8192, 16384, n//16,
                                           -int(np.log2(n)), eps_int, (-13) - E_NORM - 15])
            return np.frombuffer(bytes(sp[16384:16384+2*n]), dtype="<i2").astype(np.int64)
        w  = norm_w(name)
        v  = m.astype(f32)*f32(np.exp2(e))
        ss = fsum32(v*v)
        r  = f32(1.0)/np.sqrt(f32(f32(ss/f32(len(v))) + f32(h["rms_eps"])))
        o  = v*r*w*f32(np.exp2(-E_NORM))
        return np.clip(lroundf(o), -32767, 32767)

    def quant8_sp(m):
        # Scale to the int8 range instead of shifting by a fixed 8. After rmsnorm the mantissas
        # sit near 2**11, so >>8 handed the matvec int8 values of about +-9 - 3 of the 7 bits
        # available - and cost ~30% rel-rms on every matvec output, compounding layer over layer
        # into garbage tokens. bit_length()-7 is clip-free by construction (amax>>sh < 128).
        amax = int(np.abs(m).max()) if len(m) else 0
        sh   = max(0, amax.bit_length() - 7)
        return np.clip(bx.rne_shift(m, sh), -127, 127).astype(np.int8), sh

    def norm_i32_sp(res):
        amax = int(np.abs(res).max()) if len(res) else 0
        k = 0
        while (amax >> k) > 16383:
            k += 1
        return np.clip(res.astype(np.int64) >> k, -32767, 32767), k

    def silu_sp(gm, eg, um, eu):
        if modes.get("silu_mul", "cpu") == "hw":
            n  = len(gm)
            sp = bytearray(4*((2*n + 2048)//2048 + 2)*2048)
            sp[0:2*n]         = gm.astype("<i2").tobytes()
            sp[8192:8192+2*n] = um.astype("<i2").tobytes()
            words = vasm.assemble(vasm.PROGRAMS["silu_mul"]())
            vasm.VecInterp(sp).run(words, [0, 8192, 16384, n//16, -(eg + 6), 0, -15])
            return np.frombuffer(bytes(sp[16384:16384+2*n]), dtype="<i2").astype(np.int64)
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

    n_layer, n_embd = h["n_layer"], h["n_embd"]
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

    def res_rmsnorm_head(m, es, w16):
        n  = len(m)
        sp = bytearray(3*4096)
        sp[0:2*n]         = np.asarray(m).astype("<i2").tobytes()
        sp[4096:4096+2*n] = np.asarray(w16).astype("<i2").tobytes()
        eps_int = max(1, int(round(float(h["rms_eps"])*2.0**(-2*es))))
        vec_prog("rmsnorm", sp, [0, 4096, 8192, n//16, -int(np.log2(n)), eps_int,
                                 (-13) - E_NORM - 15])
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
        spb = bytearray(8192)
        spb[0:4*len(buf)] = buf.astype("<i4").tobytes()
        lg = int(np.log2(hd))
        Ceff  = 523 if (lg & 1) else 739
        sigma = eq - Bk + Rk - (lg >> 1)
        s1v = 11
        words = vasm.assemble(vasm.PROGRAMS["softmax8"]())
        S, _ = vasm.VecInterp(spb).run(words, [0, Ceff, 4096, len(buf)//16,
                                               -s1v, sigma + 6 + s1v])
        p8 = np.frombuffer(bytes(spb[4096:4096+len(buf)]), dtype=np.int8).astype(np.int64)
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
        spb  = bytearray(8192)
        spb[0:4*len(buf)] = buf.astype("<i4").tobytes()
        lg = int(np.log2(hd))
        Ceff  = 523 if (lg & 1) else 739
        sigma = eq - Bk + Rk - (lg >> 1)
        s1v   = 11
        words = vasm.assemble(vasm.PROGRAMS["softmax8"]())
        S, _  = vasm.VecInterp(spb).run(words, [0, Ceff, 4096, len(buf)//16,
                                                -s1v, sigma + 6 + s1v])
        p8 = np.frombuffer(bytes(spb[4096:4096+len(buf)]), dtype=np.int8).astype(np.int64)
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

    def kv_hbm_scores(l, kvh, qh):
        eq = 0
        amax = np.abs(qh).max()
        if amax > 0:
            eq = int(np.frexp(f32(amax))[1]) - 7
        q8 = np.clip(lroundf(qh*f32(np.exp2(-eq))), -127, 127).astype(np.int8)
        rowsd  = hbm_kv[(l, kvh, 0)]
        ekmax  = max(e for e, _ in rowsd)
        B      = 22 - ekmax
        bound  = (hd//32)*32*127*127*1024 << 12
        R      = 0
        while (bound >> R) >= (1 << 31):
            R += 1
        scales = np.array([[2.0**e]*(hd//32) for e, _ in rowsd], dtype=np.float16)
        codes  = np.stack([c for _, c in rowsd])
        res    = bx.matvec_q8(scales, codes, q8, B, R)
        fs     = f32(np.ldexp(f32(1.0/np.sqrt(hd)), eq - B + R))
        return res.astype(f32)*fs

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
        for l in range(n_layer):
            p = f"blk.{l}."
            # Attention block.
            hm = rmsnorm_sp(xm, ex, p + "attn_norm.weight")
            x8, sh8 = quant8_sp(hm)
            e8 = E_NORM + sh8
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
                _, rtab = img.tensor("rope_tab")
                trow = rtab[pos]
                qh8 = []
                for i2 in range(n_head):
                    mh = res_rmsnorm_head(qm[i2*hd:(i2+1)*hd], eqr, qn16.reshape(-1))
                    mh = res_rope_head(mh, trow)
                    q_i8, q_sh = quant8_sp(mh)
                    qh8.append((q_i8, E_NORM + q_sh))
                for i2 in range(n_kv):
                    mh = res_rmsnorm_head(km[i2*hd:(i2+1)*hd], ekr, kn16.reshape(-1))
                    mh = res_rope_head(mh, trow)
                    # Per-row exponents: matvec_q8's headroom depends on the spread of the row
                    # exponents, not their absolute value, so scaled rows keep the same bound.
                    k_i8, k_sh = quant8_sp(mh)
                    hbm_kv.setdefault((l, i2, 0), []).append(
                        (E_NORM + k_sh, k_i8.astype(np.int8)))
                    v_i8, v_sh = quant8_sp(vm[i2*hd:(i2+1)*hd])
                    hbm_kv.setdefault((l, i2, 1), []).append((evr + v_sh, v_i8.astype(np.int8)))
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
                res, er = matvec_i8(p + "attn_output.weight", a8, ea8)
                dm, kk  = norm_i32_sp(res)
                xm, ex  = resid_sp(xm, ex, dm, er + kk)
                hm = rmsnorm_sp(xm, ex, p + "ffn_norm.weight")
                x8, sh8 = quant8_sp(hm)
                gres, ger = matvec_i8(p + "ffn_gate.weight", x8, E_NORM + sh8)
                gm, kg = norm_i32_sp(gres)
                ures, uer = matvec_i8(p + "ffn_up.weight", x8, E_NORM + sh8)
                um, ku = norm_i32_sp(ures)
                sm = silu_sp(gm, ger + kg, um, uer + ku)
                s8, shs = quant8_sp(sm)
                dres, der = matvec_i8(p + "ffn_down.weight", s8,
                                      (ger + kg) + (uer + ku) + 15 + shs)
                dm, kd = norm_i32_sp(dres)
                xm, ex = resid_sp(xm, ex, dm, der + kd)
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
