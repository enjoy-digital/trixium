#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""bonsai_convert: GGUF (PrismML PQ2_0 / legacy Q2_0 g128 / PTQ1_0) -> BONS hardware weight image.

Subcommands:
  inspect  <gguf>                    - dump metadata + tensor table.
  convert  --gguf F --channels N --out IMG [--manifest J] [--tokenizer T] [--layers K] [--check]
  verify   --image IMG               - re-CRC all tensor payloads.

Vocab-sharded tensors (embeddings/LM head) are row-padded to divide the channel count with
zero-weight rows (code 1 -> w = 0, scale 0); the top-k unit masks rows >= vocab."""

import argparse
import hashlib
import time
import sys

import numpy as np

from trixium import layout
from trixium.layout import (
    ROLE_ATTN_QKV, ROLE_ATTN_ZGATE, ROLE_SSM_OUT, ROLE_SSM_BETA, ROLE_SSM_ALPHA,
    ROLE_SSM_CONV, ROLE_SSM_A, ROLE_SSM_DT, ROLE_SSM_NORM, ROLE_HADAMARD,
    KIND_TERNARY, KIND_Q8, KIND_F32, FLAG_REPLICATED,
    ROLE_TOK_EMB, ROLE_OUT_NORM, ROLE_LM_HEAD,
    ROLE_ATTN_NORM, ROLE_ATTN_Q, ROLE_ATTN_K, ROLE_ATTN_V, ROLE_ATTN_OUT,
    ROLE_ATTN_Q_NORM, ROLE_ATTN_K_NORM,
    ROLE_FFN_NORM, ROLE_FFN_GATE, ROLE_FFN_UP, ROLE_FFN_DOWN,
    ROLE_MTP_EH, ROLE_MTP_ENORM, ROLE_MTP_HNORM, ROLE_MTP_HEAD_NORM,
)
from trixium.model import qformats
from trixium.model.gguf_reader import GGUFReader, inspect as gguf_inspect
from trixium.model.golden import config_from_gguf, check_engine_support
from trixium.model.bons import TensorSpec, write_image, BONSReader, decode_rows

# GGUF name -> (role, replicated) ------------------------------------------------------------------

_GLOBAL_ROLES = {
    "token_embd.weight"  : (ROLE_TOK_EMB,  False),
    "output_norm.weight" : (ROLE_OUT_NORM, True),
    "output.weight"      : (ROLE_LM_HEAD,  False),
}
_BLK_ROLES = {
    "attn_norm"   : (ROLE_ATTN_NORM,   True),
    "attn_q"      : (ROLE_ATTN_Q,      False),
    "attn_k"      : (ROLE_ATTN_K,      False),
    "attn_v"      : (ROLE_ATTN_V,      False),
    "attn_q_norm" : (ROLE_ATTN_Q_NORM, True),
    "attn_k_norm" : (ROLE_ATTN_K_NORM, True),
    "attn_output" : (ROLE_ATTN_OUT,    False),
    "ffn_norm"    : (ROLE_FFN_NORM,    True),
    "ffn_gate"    : (ROLE_FFN_GATE,    False),
    "ffn_up"      : (ROLE_FFN_UP,      False),
    "ffn_down"    : (ROLE_FFN_DOWN,    False),
    # The 27B's hybrid layers (Qwen3.5 names in the GGUF).
    "post_attention_norm" : (ROLE_FFN_NORM,   True),   # the FFN's input norm, renamed
    "attn_qkv"    : (ROLE_ATTN_QKV,    False),
    "attn_gate"   : (ROLE_ATTN_ZGATE,  False),
    "ssm_out"     : (ROLE_SSM_OUT,     False),
    "ssm_beta"    : (ROLE_SSM_BETA,    False),
    "ssm_alpha"   : (ROLE_SSM_ALPHA,   False),
    "ssm_conv1d"  : (ROLE_SSM_CONV,    True),
    "ssm_a"       : (ROLE_SSM_A,       True),
    "ssm_dt"      : (ROLE_SSM_DT,      True),
    "ssm_norm"    : (ROLE_SSM_NORM,    True),
    # The MTP head (Qwen3.5 nextn block, --mtp): its decoder block uses the names above.
    "nextn.eh_proj"          : (ROLE_MTP_EH,        False),
    "nextn.enorm"            : (ROLE_MTP_ENORM,     True),
    "nextn.hnorm"            : (ROLE_MTP_HNORM,     True),
    "nextn.shared_head_norm" : (ROLE_MTP_HEAD_NORM, True),
}

_DN_LAYER_TENSORS = ("attn_norm", "attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha", "ssm_conv1d",
                     "ssm_a", "ssm_dt", "ssm_norm", "ssm_out", "post_attention_norm",
                     "ffn_gate", "ffn_up", "ffn_down")
_ATTN_LAYER_TENSORS_Q35 = ("attn_norm", "attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm",
                           "attn_output", "post_attention_norm", "ffn_gate", "ffn_up", "ffn_down")
_MTP_TENSORS = _ATTN_LAYER_TENSORS_Q35 + ("nextn.eh_proj", "nextn.enorm", "nextn.hnorm",
                                          "nextn.shared_head_norm")

def _name_role(name):
    if name in _GLOBAL_ROLES:
        return 0, *_GLOBAL_ROLES[name]
    if name.startswith("blk."):
        _, l, base = name.split(".", 2)
        base = base.removesuffix(".weight").removesuffix(".bias")
        role, repl = _BLK_ROLES[base]
        return int(l), role, repl
    raise KeyError(name)

# Conversion ---------------------------------------------------------------------------------------

def _hadamard_specs(cfg):
    """PrismML's rotation as ternary tensors, one per (input width, block): rows = cols =
    block, row i = R[i] * s[b*block + i], scale 1/sqrt(block) (exact in fp16 for a power-of-
    two block). The runtime computes x' blockwise: block b of x' = tensor (w, b) against
    x[b*block : (b+1)*block] - one engine matvec per block with the x slice as its input,
    the tensor sharded over the channels like any other. Entry layer = width index * 256 +
    block index (layout.hadamard_layer); the widths are listed in the header."""
    from trixium.model.golden import hadamard_matrix
    block = cfg.hadamard_block
    R = hadamard_matrix(block)*np.sqrt(float(block))          # +-1
    specs = []
    for wi, width in enumerate(sorted(cfg.hadamard_signs)):
        s = np.asarray(cfg.hadamard_signs[width], dtype=np.float32)
        for b in range(width//block):
            codes  = (R*s[b*block:(b + 1)*block][None, :] + 1).astype(np.uint8)
            scales = np.full((block, block//layout.GROUP_TERNARY), 1.0/np.sqrt(float(block)), np.float16)
            specs.append(TensorSpec(f"had.{width}.{b}", ROLE_HADAMARD, layout.hadamard_layer(wi, b),
                                    KIND_TERNARY, block, block, scales=scales, values=codes, flags=0))
    return specs

def _spec_from_gguf(reader, name, n_channels, vocab_pad):
    t     = reader.tensors[name]
    layer, role, repl = _name_role(name)
    flags = FLAG_REPLICATED if repl else 0

    if t.type_id in (qformats.GGML_Q2_0, qformats.GGML_PQ2_0, qformats.GGML_PTQ1_0,
                     qformats.GGML_Q1_0):
        rows, cols = t.shape
        raw = reader.tensor_bytes(name)
        # Scales now (bias/rshift need them: a few MB for the whole model); the codes
        # through a loader when the writer gets to this tensor, one tensor in memory at
        # a time (the 1.7B's codes alone are 1.7 GB).
        if t.type_id == qformats.GGML_Q2_0 and qformats.q2_0_group(raw, rows*cols) != 128:
            # PrismML gave type 42 to upstream's group-64 Q2_0 in 2026-06 (the *_Q2_0_g64
            # files); two scales per 128 weights do not fit the engine's g128 rows.
            raise NotImplementedError(f"{name}: group-64 Q2_0 - convert the PQ2_0 (g128) "
                                      "file of the same model instead")
        if t.type_id in (qformats.GGML_Q2_0, qformats.GGML_PQ2_0):
            scales = qformats.q2_0_scales(raw)
            unpack = lambda: qformats.q2_0_unpack(raw)[1]
        elif t.type_id == qformats.GGML_PTQ1_0:
            # Dense trits (5 per byte): same weights as PQ2_0, decoded to the same codes.
            scales = qformats.ptq1_0_scales(raw)
            unpack = lambda: qformats.ptq1_0_unpack(raw)[1]
        else:
            # 1-bit builds ("Bonsai" as opposed to "Ternary-Bonsai"): run on the ternary
            # engine via code expansion (see qformats.q1_0_to_ternary_codes).
            scales = qformats.q1_0_to_ternary_codes(raw)[0]
            unpack = lambda: qformats.q1_0_to_ternary_codes(raw)[1]
        scales = scales.reshape(rows, cols//layout.GROUP_TERNARY)
        pad = 0
        if not repl and rows % n_channels != 0:
            assert role in (ROLE_TOK_EMB, ROLE_LM_HEAD), name
            pad    = vocab_pad - rows
            scales = np.concatenate([scales, np.zeros((pad, scales.shape[1]), np.float16)])
        def load(unpack=unpack, rows=rows, cols=cols, pad=pad):
            codes = unpack().reshape(rows, cols)
            if pad:
                codes = np.concatenate([codes, np.ones((pad, cols), np.uint8)]) # code 1 -> w=0.
            return codes
        return TensorSpec(name, role, layer, KIND_TERNARY, rows + pad, cols,
                          scales=scales, loader=load, flags=flags)

    if t.type_name == "F32":
        values = qformats.f32_dequant(reader.tensor_bytes(name)).reshape(t.shape)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if role == ROLE_SSM_CONV:
            # (channels, taps): stored as rows of 1024 floats, tap i of channel c at flat
            # index c*taps + i, so the firmware indexes a beat-aligned table.
            values = values.reshape(-1, 1024)
        rows, cols = values.shape
        return TensorSpec(name, role, layer, KIND_F32, rows, cols, values=values,
                          flags=FLAG_REPLICATED)

    if t.type_name == "Q8_0":
        # Q8_0 is the engine's Q8 row format as it is: one fp16 scale per 32 int8 weights
        # (the MTP head at Q8, --mtp).
        rows, cols = t.shape
        raw = np.frombuffer(reader.tensor_bytes(name), dtype=np.uint8).reshape(rows, cols//32, 34)
        scales = np.ascontiguousarray(raw[:, :, :2]).view(np.float16).reshape(rows, cols//32)
        values = np.ascontiguousarray(raw[:, :, 2:]).view(np.int8).reshape(rows, cols)
        return TensorSpec(name, role, layer, KIND_Q8, rows, cols, scales=scales.copy(),
                          values=values.copy(), flags=flags)

    if t.type_name == "BF16":
        # The DeltaNet head projections (beta/alpha: value heads x n_embd): Q8 rows for the
        # engine, one fp16 scale per 32 weights, rounded to nearest.
        assert role in (ROLE_SSM_BETA, ROLE_SSM_ALPHA), name
        rows, cols = t.shape
        raw = np.frombuffer(reader.tensor_bytes(name), dtype=np.uint16).astype(np.uint32) << 16
        w = raw.view(np.float32).reshape(rows, cols)
        g = w.reshape(rows, cols//layout.GROUP_Q8, layout.GROUP_Q8)
        amax = np.abs(g).max(axis=2)
        scales = (amax/127.0).astype(np.float16)
        sc = scales.astype(np.float32)[:, :, None]
        q = np.where(sc > 0, np.round(g/np.where(sc > 0, sc, 1.0)), 0).clip(-127, 127)
        return TensorSpec(name, role, layer, KIND_Q8, rows, cols, scales=scales,
                          values=q.reshape(rows, cols).astype(np.int8), flags=0)

    raise NotImplementedError(f"{name}: unsupported tensor type {t.type_name}")

def convert(gguf_path, n_channels, out_path, manifest_path=None, tokenizer_path=None,
            max_layers=None, check=False, emb16=True, chan_stride=None, weight_base=None,
            window_reserve=0, kseg_max=layout.KSEG_MAX, mtp_path=None):
    """mtp_path: an MTP-only GGUF (tools/speculative/mk_rotated.py + llama-quantize): its
    blk.<n_layer>.* decoder block and nextn tensors go in at layer n_layer; its embedding,
    LM head and output norm are not used (the head shares the model's)."""
    reader = GGUFReader(gguf_path)
    cfg    = config_from_gguf(reader)
    cfg.mtp_layers = 0
    try:
        check_engine_support(cfg)
    except SystemExit as e:
        # The image is still built: the firmware refuses at boot what it cannot run, and
        # the tools (the mirrors, the golden) read the image meanwhile.
        print(f"  NOTE: {e}")
    if max_layers is not None:
        cfg.n_layer = min(cfg.n_layer, max_layers)
        cfg.name    = f"{cfg.name}-l{cfg.n_layer}"
    vocab_pad = cfg.vocab_padded(n_channels)

    # Canonical tensor order: embeddings, per-layer blocks, final norm, head.
    names = ["token_embd.weight"]
    for l in range(cfg.n_layer):
        if cfg.layer_type(l) == "deltanet":
            bases = _DN_LAYER_TENSORS
        elif f"blk.{l}.post_attention_norm.weight" in reader.tensors:
            bases = _ATTN_LAYER_TENSORS_Q35
        else:
            bases = ("attn_norm", "attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm",
                     "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down")
        for base in bases:
            suffix = ".bias" if base == "ssm_dt" else ("" if base == "ssm_a" else ".weight")
            names.append(f"blk.{l}.{base}{suffix}")
    names.append("output_norm.weight")
    if "output.weight" in reader.tensors:
        names.append("output.weight")

    specs = [_spec_from_gguf(reader, name, n_channels, vocab_pad) for name in names]
    if mtp_path:
        mreader = GGUFReader(mtp_path)
        L = cfg.n_layer
        mnames = [f"blk.{L}.{base}.weight" for base in _MTP_TENSORS]
        missing = [n for n in mnames if n not in mreader.tensors]
        if missing:
            raise SystemExit(f"{mtp_path}: no MTP block at layer {L} ({missing[0]} missing)")
        specs += [_spec_from_gguf(mreader, name, n_channels, vocab_pad) for name in mnames]
        cfg.mtp_layers = 1
    if cfg.hadamard_block:
        specs += _hadamard_specs(cfg)
    # Matvecs wider than one engine op (the 27B's FFN down projection, K = 17408) become
    # K-segments the firmware sums (layout.FLAG_KSEG).
    from trixium.model.bons import split_wide
    specs = [seg for spec in specs for seg in split_wide(spec, kseg_max)]
    nseg = sum(1 for sp in specs if sp.flags & layout.FLAG_KSEG)
    if nseg:
        print(f"  {nseg} K-segments (matvecs wider than {kseg_max})")

    # Does it fit the card? Each engine channel owns HBM_CHAN_STRIDE (256 MB) from
    # HBM_WEIGHT_BASE's offset into its pseudo-channel (64 MB), and the KV cache sits above
    # the weights (kv rows of 1 + head_dim/32 beats, max_ctx/n_channels per (layer, kv
    # head, K|V)). The 8B (136 MB/channel of ternary) fits; its int16 embedding table
    # (1.2 GB) does not, so that is dropped with a note rather than refused; a model whose
    # weights alone overflow is refused here instead of by the loader (2026-09-18).
    # The window is a property of the board target (chan_stride / weight_base are what its
    # SoC exports as BONSAI_CHAN_STRIDE / BONSAI_CHAN_WEIGHT_BASE); the FK33's constants are
    # the default and the XCU1525 uses the same ones per controller, so the images are
    # shared. A target with a bigger stride passes it (--chan-stride / --weight-base).
    from trixium.model.bons import _tensor_beats
    from trixium.layout import HBM_CHAN_STRIDE, HBM_WEIGHT_BASE
    if chan_stride is None: chan_stride = HBM_CHAN_STRIDE
    if weight_base is None: weight_base = HBM_WEIGHT_BASE
    window   = chan_stride - (weight_base % chan_stride) - 0x1000 - window_reserve  # The last 4 KB: the
    # firmware's DDR round-trip test (ddr.c) writes them on boards with several controllers.
    kv_rb    = 1 + cfg.head_dim//32
    kv_bytes = (cfg.n_layer + cfg.mtp_layers)*cfg.n_kv_head*2*((cfg.max_ctx + n_channels - 1)//n_channels)*kv_rb*32
    dn_bytes = layout.deltanet_state_bytes_per_channel(cfg, n_channels)
    weights  = sum(_tensor_beats(t, n_channels) for t in specs)*32
    emb16_b  = 0
    if emb16:
        erows   = (vocab_pad + 15)//16
        emb16_b = (vocab_pad*cfg.n_embd*2 + erows*32)//n_channels   # int16 rows + exponents
    def mb(b): return f"{b/1e6:.0f} MB"
    reserved = weights + kv_bytes + dn_bytes
    if reserved > window:
        raise SystemExit(f"{cfg.name}: {mb(weights)} of weights + {mb(kv_bytes)} of KV"
                         f"{' + ' + mb(dn_bytes) + ' of DeltaNet state' if dn_bytes else ''} per "
                         f"channel exceed the {mb(window)} channel window at {n_channels} "
                         f"channels - this model does not fit the card")
    if emb16 and reserved + emb16_b > window:
        print(f"  int16 embedding table ({mb(emb16_b)}/channel) would overflow the channel "
              f"window ({mb(reserved)} used of {mb(window)}): left out, the firmware "
              f"decodes the ternary embedding rows itself (~2.6% per token)")
        emb16 = False
    print(f"  per channel: weights {mb(weights)} + KV {mb(kv_bytes)}"
          f"{' + DeltaNet state ' + mb(dn_bytes) if dn_bytes else ''}"
          f"{' + emb16 ' + mb(emb16_b) if emb16 else ''} of {mb(window)}")

    sha = b""
    if tokenizer_path:
        sha = hashlib.sha256(open(tokenizer_path, "rb").read()).digest()

    t0 = time.time()
    manifest = write_image(cfg, specs, n_channels, out_path,
                           tokenizer_sha256=sha, manifest_path=manifest_path, emb16=emb16,
                           progress=lambda name: print(f"  {name}...", flush=True))
    print(f"  {time.time() - t0:.1f} s")
    print(f"image: {out_path} payload {manifest['payload_bytes_per_channel']/1e6:.1f} MB/channel"
          f" x {n_channels} channels")

    if check:
        img = BONSReader(out_path)
        for name in ("blk.0.attn_q.weight", "blk.0.ffn_gate.weight"):
            s_img, v_img = img.tensor(name)
            s_ref, v_ref = None, None
            for spec in specs:
                if spec.name == name:
                    s_ref, v_ref = spec.scales, spec.values
            assert np.array_equal(v_img, v_ref), name
            assert np.array_equal(s_img, s_ref.astype(np.float16)), name
            print(f"  check {name}: OK")
    return manifest

def verify(image_path):
    img = BONSReader(image_path)
    h   = img.header
    print(f"BONS v{h['version']}: {h['n_tensors']} tensors, {h['n_channels']} channels, "
          f"{h['payload_beats']*32/1e6:.1f} MB/channel")
    import zlib
    bad = 0
    for e in img.entries:
        crc = 0
        for c in range(h["n_channels"]):
            data = img.entry_channel_bytes(e, c)
            crc  = zlib.crc32(bytes(data), crc) & 0xFFFFFFFF
        ok = crc == e["crc32"]
        bad += not ok
        if not ok:
            print(f"  CRC MISMATCH {e['name']}: {crc:#x} != {e['crc32']:#x}")
    print("verify: OK" if bad == 0 else f"verify: {bad} bad tensors")
    return bad == 0

# CLI ----------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GGUF -> BONS converter.")
    sub    = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect")
    p.add_argument("gguf")

    p = sub.add_parser("convert")
    p.add_argument("--gguf",      required=True)
    p.add_argument("--channels",  required=True, type=int)
    p.add_argument("--out",       required=True)
    p.add_argument("--manifest",  default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--layers",    default=None, type=int, help="Truncate to first K layers.")
    p.add_argument("--check",     action="store_true")
    p.add_argument("--no-emb16",  action="store_true",
                   help="Leave out the pre-dequantized int16 embedding table (622 MB on the "
                        "1.7B; the firmware decodes rows itself, ~2.6%% slower per token).")
    p.add_argument("--chan-stride", default=None, type=lambda x: int(x, 0),
                   help="The target's per-channel stride (BONSAI_CHAN_STRIDE; default: the FK33's 256 MB).")
    p.add_argument("--kseg-max", default=layout.KSEG_MAX, type=int,
                   help="Split matvecs wider than this into K-segments (default: the engine op's "
                        "limit; small values exercise the path on tiny models).")
    p.add_argument("--window-reserve", default=0, type=lambda x: int(x, 0),
                   help="Bytes the last window of a controller loses to the per-channel bank skew "
                        "((chan_mod-1)*BONSAI_CHAN_SKEW: 96 KB on the XCU1525).")
    p.add_argument("--weight-base", default=None, type=lambda x: int(x, 0),
                   help="The target's weight base (BONSAI_CHAN_WEIGHT_BASE; default: the FK33's).")
    p.add_argument("--mtp", default=None,
                   help="MTP-only GGUF whose head goes in at layer n_layer (speculative decoding).")

    p = sub.add_parser("verify")
    p.add_argument("--image", required=True)

    args = parser.parse_args()
    if args.cmd == "inspect":
        gguf_inspect(args.gguf)
    elif args.cmd == "convert":
        manifest = args.manifest or args.out.rsplit(".", 1)[0] + ".json"
        convert(args.gguf, args.channels, args.out, manifest, args.tokenizer,
                args.layers, args.check, emb16=not args.no_emb16,
                chan_stride=args.chan_stride, weight_base=args.weight_base,
                window_reserve=args.window_reserve, kseg_max=args.kseg_max, mtp_path=args.mtp)
    elif args.cmd == "verify":
        sys.exit(0 if verify(args.image) else 1)

if __name__ == "__main__":
    main()
