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
def test_targets_use_one_cluster_implementation():
    """The former source-copy guard is replaced by a shared construction boundary.

    Functional seam/reset/store tests live in test_integration.py; these checks
    prevent a target from silently reinstating its own divergent third copy.
    Internal Migen names are allowed to change, but require fresh physical gates.
    """
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for fname in ("trixium/targets/fk33.py", "trixium/targets/xcu1525.py", "trixium/targets/sim.py"):
        tree = ast.parse((root / fname).read_text())
        calls = [node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        assert calls.count("add_bonsai_cluster") == 1, fname
        for constructor in ("BonsaiMatVecEngine", "BonsaiVectorUnit", "BonsaiSequencer",
                            "BonsaiEngineCSR", "BonsaiHBMLoader", "BonsaiEngineResetDomain"):
            assert constructor not in calls, (fname, constructor)

    portable = (root / "trixium/core/core.py").read_text()
    assert "vecunit.seq_seed7.eq(seq.vec_seed7)" in portable
    assert "platform.add_ip" not in portable
    assert "add_platform_command" not in portable
