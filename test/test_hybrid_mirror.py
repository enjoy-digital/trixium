#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Structural Python reference for firmware/hybrid.c - the hybrid models' CPU path.

hybrid27_generate walks a BONS v3 image: every matvec through the
engine's bit-exact model (int8 input with a power-of-two scale, int32 results, float
scaling), with sequential float32 sums for the other operations. The default SiLU
models the RV32F baseline's target-libm LUT, fused interpolation and polynomial tails.
exact=True retains direct mathematical SiLU for comparison with the FP32 golden.
Other transcendental operations still use NumPy; passing the pinned tiny trace does
not prove bit-exactness for arbitrary models, inputs or target libm implementations.
It is what tools/check_sim_tokens.py compares the sim's GEN legs against when the image is
a hybrid, and the test below pins it to the fp32 golden on the tiny hybrid model.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trixium import layout
from trixium.model import bitexact as bx
from trixium.model.bons import BONSReader

f32 = np.float32


def _fsum32(v):
    acc = f32(0.0)
    for e in np.asarray(v, dtype=f32):
        acc = f32(acc + e)
    return acc


def _sumsq64(v):
    """The canonical sum of squares (docs/canonical-reductions.md, hybrid.c sumsq64)."""
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
    fast=True: the DeltaNet recurrence in the fast path's arithmetic (docs/xcu1525.md) -
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
    n_layer, n_embd, ffn_dim = h["n_layer"], h["n_embd"], h["ffn_dim"]
    n_head, n_kv, hd = h["n_head"], h["n_kv_head"], h["head_dim"]
    nk, hk, nv, hv = h["dn_num_k_heads"], h["dn_k_head_dim"], h["dn_num_v_heads"], h["dn_v_head_dim"]
    nq, nvh, K = nk*hk, nv*hv, h["dn_conv_kernel"]
    rope_n = h["rope_dims"] or hd
    eps = f32(h["rms_eps"])
    B = h["hadamard_block"]
    widths = list(h["hadamard_widths"])
    gated = bool(h["attn_gated"])
    q_dim, kv_dim, n_rep = n_head*hd, n_kv*hd, n_head//n_kv

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
        return (x/nrm).astype(f32)

    def sigmoid(x): return (f32(1.0)/(f32(1.0) + np.exp(-x, dtype=f32))).astype(f32)
    if silu_mode == "rv32f":
        from trixium.model.rv32_math import sigmoid_rv32 as sigmoid   # canonical LUT sigmoid
    if silu_mode == "rv32f":
        from trixium.model.rv32_math import silu_rv32 as silu
    else:
        def silu(x): return (x/(f32(1.0) + np.exp(-x, dtype=f32))).astype(f32)
    def softplus(x): return np.where(x > 20, x, np.log1p(np.exp(x, dtype=f32), dtype=f32)).astype(f32)

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
        """One head in the two-pass vector-unit arithmetic (docs/xcu1525.md): the state
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
            decay = f32(np.exp(g[hh], dtype=f32))
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
            ex = np.exp((sc - mx).astype(f32), dtype=f32)
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


def test_mirror_agrees_with_the_fp32_golden_on_the_tiny_hybrid():
    """The mirror's structure against the fp32 golden: run exact (fp32 weights, no activation
    quantization) it must reproduce the golden's tokens on the tiny hybrid model, which pins
    the recurrence, the head order, the gates and the rotation independently of the int8
    noise. The sim then compares the firmware to the quantized mirror exactly."""
    import tempfile
    from trixium.model.configs import BONSAI_TINY_HYBRID
    from trixium.model.tinygen import generate, make_tiny_hybrid_tensors
    from trixium.model.golden import FloatModel
    cfg = BONSAI_TINY_HYBRID
    with tempfile.TemporaryDirectory() as d:
        img_path, _, _ = generate(d, 2, 1234, cfg=cfg)
        img = BONSReader(img_path)
        got = hybrid27_generate(img, [1, 2, 3], 4, exact=True)
        events = []
        def trace(pos, layer, tag, dtype, values):
            events.append((pos, layer, tag, dtype, np.asarray(values).size))
        quant = hybrid27_generate(img, [1, 2, 3], 4, trace=trace)
        # Match the firmware's contract, including NO logits on early prefill forwards
        # and NO unused forward after the last output token.
        assert len(events) == 6*(1 + 9*cfg.n_layer) + 4*3
        assert sorted({e[0] for e in events}) == list(range(6))
        assert [(p, n) for p, l, tag, dtype, n in events if tag == "argmax"] == [
            (p, 1) for p in range(2, 6)]
        for p in range(6):
            for l in range(cfg.n_layer):
                tags = [tag for pos, layer, tag, dtype, n in events if (pos, layer) == (p, l)]
                assert tags == ["attn_norm", "dn_out" if cfg.layer_type(l) == "deltanet" else "attn_out",
                                "attn_res", "ffn_norm", "ffn_gate", "ffn_up", "ffn_act",
                                "ffn_down", "ffn_res"]
    specs = make_tiny_hybrid_tensors(cfg, 1234)
    t = {}
    for s in specs:
        if s.kind == layout.KIND_TERNARY:
            w = (s.get_values().astype(np.float32).reshape(s.rows, -1, 128) - 1.0)*s.scales.astype(np.float32)[:, :, None]
            t[s.name] = w.reshape(s.rows, s.cols)
        elif s.kind == layout.KIND_Q8:
            w = s.get_values().astype(np.float32).reshape(s.rows, -1, 32)*s.scales.astype(np.float32)[:, :, None]
            t[s.name] = w.reshape(s.rows, s.cols)
        elif s.kind == layout.KIND_F32:
            v = s.get_values()
            t[s.name] = v.reshape(-1, cfg.dn_conv_kernel) if s.role == layout.ROLE_SSM_CONV else v.reshape(-1)
    m = FloatModel(cfg, t)
    want = m.generate([1, 2, 3], 4)
    assert got == want, (got, want)
    # The quantized mirror (what the firmware computes) agrees on the first token at least;
    # later tokens can flip on this random model's small logit margins (0.03 at step 1).
    assert quant[0] == want[0], (quant, want)


def test_rv32_profile_closes_archived_fpu_trace(tmp_path):
    """A compiled firmware baseline, not a trace generated by this Python model: the
    VexiiRiscv RV32F SoC simulation's typed trace of tiny-hybrid-1k (ids 1 2 3), archived
    from the canonical-numerics sim gate (docs/canonical-reductions.md). The mirror's rv32f
    profile must reproduce every checkpoint; the mathematical profile must not."""
    from pathlib import Path
    from trixium.model.configs import BONSAI_TINY_HYBRID_1K
    from trixium.model.tinygen import generate
    from tools.card_vs_mirror import format_typed_trace, parse
    image, _, _ = generate(str(tmp_path), 8, 1234, cfg=BONSAI_TINY_HYBRID_1K)
    expected, order = parse(Path(__file__).parent / 'data/tiny-hybrid-1k-fpu-trace.txt')
    for mode, full in [('rv32f', True), ('mathematical', False)]:
        lines = []
        def trace(pos, layer, tag, dtype, values):
            lines.append(format_typed_trace(pos, layer, tag, dtype, values))
        hybrid27_generate(BONSReader(image), [1, 2, 3], 1, trace=trace, silu_mode=mode)
        output = tmp_path / f'{mode}.txt'
        output.write_text('\n'.join(lines) + '\n')
        actual, actual_order = parse(output)
        assert actual_order == order
        same = sum(actual[k].signature == expected[k].signature for k in order)
        assert (same == len(order)) == full, (mode, same, len(order))
