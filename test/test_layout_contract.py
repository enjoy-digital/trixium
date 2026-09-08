#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

# Layout/BONS contract tests: beat math invariants, header/entry roundtrip, and a full
# tiny-image write/read roundtrip (every tensor recovered bit-exactly from its shards).

import numpy as np

from trixium import layout
from trixium.model import bons
from trixium.model.configs import BONSAI_TINY, BONSAI_1P7B, BONSAI_8B
from trixium.model.tinygen import make_tiny_tensors, generate

def test_beat_math():
    # Ternary: one code beat per 128-group.
    assert layout.code_beats(4096, layout.KIND_TERNARY) == 32
    assert layout.scale_beats(4096, layout.KIND_TERNARY) == 2   # 32 scales -> 64B -> 2 beats.
    assert layout.row_beats(2048, layout.KIND_TERNARY) == 1 + 16
    # Q8: one beat per 32-group (32 int8).
    assert layout.code_beats(2048, layout.KIND_Q8) == 64
    assert layout.scale_beats(2048, layout.KIND_Q8) == 4        # 64 scales -> 128B -> 4 beats.

def test_model_dims_divide():
    # Every sharded tensor's rows must divide the max channel count (vocab via padding),
    # every ternary K the 128 group.
    for cfg in (BONSAI_1P7B, BONSAI_8B):
        for rows in (cfg.q_dim, cfg.kv_dim, cfg.n_embd, cfg.ffn_dim, cfg.vocab_padded(32)):
            assert rows % 32 == 0, (cfg.name, rows)
        for cols in (cfg.n_embd, cfg.q_dim, cfg.ffn_dim):
            assert cols % layout.GROUP_TERNARY == 0, (cfg.name, cols)
    assert BONSAI_1P7B.vocab_padded(32) == 151680 # 151669 padded.
    cfg = BONSAI_TINY
    for rows in (cfg.q_dim, cfg.kv_dim, cfg.n_embd, cfg.ffn_dim, cfg.vocab):
        assert rows % 2 == 0
    for cols in (cfg.n_embd, cfg.q_dim, cfg.ffn_dim):
        assert cols % layout.GROUP_TERNARY == 0

def test_header_roundtrip():
    hdr = layout.pack_header(BONSAI_TINY, n_channels=2, n_tensors=25, payload_beats=1000,
                             table_crc32=0xDEADBEEF, tokenizer_sha256=b"\x01"*32)
    assert len(hdr) == layout.BONS_HEADER_SIZE
    h = layout.unpack_header(hdr)
    assert h["n_layer"] == BONSAI_TINY.n_layer
    assert h["n_channels"] == 2
    assert h["payload_beats"] == 1000
    assert h["table_crc32"] == 0xDEADBEEF
    assert h["tokenizer_sha256"] == b"\x01"*32

def test_entry_roundtrip():
    e = layout.pack_entry(layout.ROLE_FFN_GATE, layout.KIND_TERNARY, 0, 5, 512, 256,
                          123, 456, 0xCAFEBABE, bias=26, rshift=7,
                          name="blk.5.ffn_gate.weight")
    assert len(e) == layout.BONS_ENTRY_SIZE
    d = layout.unpack_entry(e)
    assert d["role"] == layout.ROLE_FFN_GATE
    assert d["layer"] == 5
    assert d["rows"] == 512 and d["cols"] == 256
    assert d["chan_base_beats"] == 123 and d["chan_len_beats"] == 456
    assert d["bias"] == 26 and d["rshift"] == 7
    assert d["name"] == "blk.5.ffn_gate.weight"

def test_tiny_image_roundtrip(tmp_path):
    img, man, manifest = generate(str(tmp_path), n_channels=2, seed=99)
    reader  = bons.BONSReader(img)
    tensors = make_tiny_tensors(BONSAI_TINY, seed=99) # Same seed -> same arrays.

    # write_image bakes in derived entries on top of the source list: the rope table (when the
    # config has a context and head dim) and an int16 ".w16" companion for every F32 norm
    # vector, so the image legitimately carries more tensors than make_tiny_tensors returns.
    f32 = [t for t in tensors if t.kind == layout.KIND_F32]
    n_derived = len(f32) + (1 if BONSAI_TINY.max_ctx and BONSAI_TINY.head_dim else 0)
    # ...and the pre-dequantized embedding table (mantissas + exponents) for the token
    # embedding (test_embed_table.py covers its numerics).
    emb = [t for t in tensors if t.role == layout.ROLE_TOK_EMB and t.kind == layout.KIND_TERNARY]
    n_derived += 2*len(emb)
    assert reader.header["n_tensors"] == len(tensors) + n_derived
    assert reader.header["n_channels"] == 2
    if BONSAI_TINY.max_ctx and BONSAI_TINY.head_dim:
        assert "rope_tab" in reader.by_name
    if emb:
        assert "token_embd.bfp16" in reader.by_name and "token_embd.bfp16_exp" in reader.by_name
        e16 = reader.by_name["token_embd.bfp16"]
        assert e16["kind"] == layout.KIND_NORM16 and e16["rows"] == emb[0].rows
        assert e16["cols"] == emb[0].cols
    for spec in f32:
        assert spec.name.replace(".weight", "") + ".w16" in reader.by_name
    for spec in tensors:
        scales, values = reader.tensor(spec.name)
        if spec.kind == layout.KIND_TERNARY:
            assert np.array_equal(values, spec.values)
            assert np.array_equal(scales, spec.scales.astype(np.float16))
        elif spec.kind == layout.KIND_Q8:
            assert np.array_equal(values, spec.values)
            assert np.array_equal(scales, spec.scales.astype(np.float16))
        elif spec.kind == layout.KIND_F32:
            assert np.array_equal(values, spec.values.astype(np.float32))
    # Payload lengths identical across channels (lockstep invariant).
    p0 = reader.channel_payload(0)
    p1 = reader.channel_payload(1)
    assert len(p0) == len(p1) == reader.header["payload_beats"]*layout.BEAT_BYTES
    # Sharded tensors differ between channels; replicated ones don't.
    assert not np.array_equal(reader.tensor_channel_bytes("token_embd.weight", 0),
                              reader.tensor_channel_bytes("token_embd.weight", 1))
    assert np.array_equal(reader.tensor_channel_bytes("blk.0.attn_norm.weight", 0),
                          reader.tensor_channel_bytes("blk.0.attn_norm.weight", 1))
def test_soc_definitions_wire_the_sequencer_alike():
    """trixium.py and sim/bonsai_sim.py must wire the vecunit/sequencer identically.

    There are two SoC definitions - hardware and Verilator sim - and every integration
    signal has to be spelled out in both. When they drift, the sim silently stops testing
    what the hardware does, which is the worst possible failure: gates pass, hardware is
    wrong, or (as happened) a sim gate fails for a reason that exists only in the sim.

    Paid for twice now. The include-guard case let bonsai_load upload to a stale base. The
    seq_seed7 case wired the S7 seed straight from vec_res6 in the sim only, which made the
    descriptor's w0[4] absmax-seed select a no-op there: the spill persisted the softmax p8
    sum instead of the engine absmax, norm_i32 took its shift from that, and every head came
    out misscaled. It cost several 12-minute Verilator runs and a wrong root-cause call
    before an XPROBE showed `am` equal to `P8` exactly.

    A grep is crude, but it is instant and it fails loudly on the one thing that went wrong.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    srcs = {
        "trixium.py":    (root/"trixium.py").read_text(),
        "sim/bonsai_sim.py":  (root/"sim/bonsai_sim.py").read_text(),
    }
    # (signal, expected right-hand side) for the vecunit<->sequencer seam.
    seam = [
        ("seq_seed7", "vec_seed7"),  # NOT vec_res6: that bypasses the w0[4] mux.
        ("seq_start", "vec_start"),
        ("seq_pc",    "vec_pc"),
    ]
    for fname, text in srcs.items():
        for sig, want in seam:
            m = re.search(rf"vecunit\.{sig}\.eq\(\s*(?:self\.)?seq\.(\w+)", text)
            assert m, f"{fname}: no wiring found for vecunit.{sig}"
            assert m.group(1) == want, (
                f"{fname}: vecunit.{sig} is wired from seq.{m.group(1)}, expected "
                f"seq.{want}. The two SoC definitions have drifted - see this test's "
                f"docstring for what that costs.")
