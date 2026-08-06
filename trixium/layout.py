#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

"""Single source of truth for the TriXium data layout contracts.

Shared by the weight converter, the golden model, the gateware generators/testbenches, the
firmware header generation and the host tools. Three contracts live here:

1. HW weight layout ("row packets"): how a quantized tensor is arranged in HBM so that one
   linear AXI burst region per channel feeds the engine.
2. BONS image format v1: the on-disk/in-HBM weight image (header + tensor table + payloads).
3. Engine CSR/descriptor contract constants (opcodes, kinds).

Geometry: one HBM pseudo-channel AXI beat is 256 bits = 32 bytes. For ternary tensors
(group=128), one beat carries exactly one group's 2-bit codes (128 codes x 2b = 32B), so one
beat feeds the 128-lane adder tree for one cycle. Row packet = all group scales (fp16, packed
16/beat) followed by all code beats, rows back-to-back:

  row r: [scale beat 0 .. scale beat S-1][code beat 0 .. code beat G-1]   (S = ceil(G/16))

Tensors are block-sharded by output row across N channels (channel c gets rows
[c*M/N, (c+1)*M/N)), with the same relative addresses in every channel, so a single
{base, len} descriptor drives all N weight streamers.
"""

import struct
import zlib

# Geometry -----------------------------------------------------------------------------------------

BEAT_BYTES    = 32   # One HBM AXI beat: 256 bits.
GROUP_TERNARY = 128  # Weights per ternary group (one fp16 scale each), one code beat per group.
GROUP_Q8      = 32   # Weights per int8 group (one fp16 scale each), one code beat per group.
SCALES_PER_BEAT = BEAT_BYTES//2 # 16 fp16 scales per beat.

HBM_NCHANNELS    = 32
HBM_CHANNEL_SIZE = 0x1000_0000 # 256 MB per pseudo-channel.

# Tensor kinds -------------------------------------------------------------------------------------

KIND_TERNARY = 0 # 2-bit codes {0,1,2}, group=128, w = (q-1)*scale.
KIND_Q8      = 1 # int8 quants, group=32, w = q*scale.
KIND_F16     = 2 # Raw fp16 rows (no scales).
KIND_F32     = 3 # Raw fp32 rows (no scales; norms etc, consumed by CPU/vector unit).
KIND_NORM16  = 4 # int16 rows at fixed exponent NORM16_EXP (norm weights for the vector unit).

NORM16_EXP = -13 # w = q * 2^-13: |w| < 4 with ~1.2e-4 resolution - covers rmsnorm gammas.

KIND_NAMES = {KIND_TERNARY: "ternary", KIND_Q8: "q8", KIND_F16: "f16", KIND_F32: "f32",
              KIND_NORM16: "norm16"}

KIND_GROUP = {KIND_TERNARY: GROUP_TERNARY, KIND_Q8: GROUP_Q8}

# Tensor flags.
FLAG_REPLICATED = 0x01 # Tensor payload identical in every channel (not row-sharded).

# Tensor roles -------------------------------------------------------------------------------------

(ROLE_TOK_EMB, ROLE_OUT_NORM, ROLE_LM_HEAD,
 ROLE_ATTN_NORM, ROLE_ATTN_Q, ROLE_ATTN_K, ROLE_ATTN_V, ROLE_ATTN_OUT,
 ROLE_ATTN_Q_NORM, ROLE_ATTN_K_NORM,
 ROLE_FFN_NORM, ROLE_FFN_GATE, ROLE_FFN_UP, ROLE_FFN_DOWN,
 ROLE_ROPE_TAB) = range(15)

ROLE_NAMES = {
    ROLE_TOK_EMB     : "tok_emb",     ROLE_OUT_NORM  : "out_norm",  ROLE_LM_HEAD  : "lm_head",
    ROLE_ATTN_NORM   : "attn_norm",   ROLE_ATTN_Q    : "attn_q",    ROLE_ATTN_K   : "attn_k",
    ROLE_ATTN_V      : "attn_v",      ROLE_ATTN_OUT  : "attn_out",  ROLE_ATTN_Q_NORM: "attn_q_norm",
    ROLE_ATTN_K_NORM : "attn_k_norm", ROLE_FFN_NORM  : "ffn_norm",  ROLE_FFN_GATE : "ffn_gate",
    ROLE_FFN_UP      : "ffn_up",      ROLE_FFN_DOWN  : "ffn_down",
    ROLE_ROPE_TAB    : "rope_tab",
}

# Row packet math ----------------------------------------------------------------------------------

def ceil_div(a, b):
    return (a + b - 1)//b

def row_groups(cols, kind):
    group = KIND_GROUP[kind]
    assert cols % group == 0, f"cols {cols} not a multiple of group {group}"
    return cols//group

def scale_beats(cols, kind):
    """Beats of fp16 group scales prepended to each row (0 for f16/f32/norm16 rows)."""
    if kind in (KIND_F16, KIND_F32, KIND_NORM16):
        return 0
    return ceil_div(row_groups(cols, kind), SCALES_PER_BEAT)

def code_beats(cols, kind):
    """Beats of packed codes/values per row."""
    if kind == KIND_TERNARY:
        return cols//GROUP_TERNARY          # 32B of 2-bit codes per 128-group = 1 beat.
    if kind == KIND_Q8:
        return cols//GROUP_Q8               # 32 int8 per 32-group = 1 beat.
    if kind == KIND_F16:
        return ceil_div(2*cols, BEAT_BYTES)
    if kind == KIND_F32:
        return ceil_div(4*cols, BEAT_BYTES)
    if kind == KIND_NORM16:
        return ceil_div(2*cols, BEAT_BYTES)
    raise ValueError(kind)

def row_beats(cols, kind):
    return scale_beats(cols, kind) + code_beats(cols, kind)

def shard_rows(rows, n_channels, channel):
    """Row range [start, count) of `channel` for a block-sharded tensor."""
    assert rows % n_channels == 0, f"rows {rows} not divisible by {n_channels} channels"
    per = rows//n_channels
    return channel*per, per

def tensor_chan_beats(rows, cols, kind, n_channels, flags=0):
    """Per-channel payload length in beats."""
    if flags & FLAG_REPLICATED:
        return rows*row_beats(cols, kind)
    assert rows % n_channels == 0, f"rows {rows} not divisible by {n_channels} channels"
    return (rows//n_channels)*row_beats(cols, kind)

# BONS image format v1 -----------------------------------------------------------------------------
#
# Single file: [header 4 KiB][tensor table, 64 B/entry, padded to 4 KiB][channel 0 payload]
# [channel 1 payload]...[channel N-1 payload]. All channel payloads have the same length
# (payload_beats). In HBM, each channel's payload is uploaded at the same base offset of its
# pseudo-channel; the header+table are uploaded at HBM_HEADER_BASE in channel 0 so the firmware
# can walk the model without host-side sideband.

BONS_MAGIC       = 0x534E4F42 # "BONS" little-endian.
BONS_VERSION     = 1
BONS_HEADER_SIZE = 4096
BONS_ENTRY_SIZE  = 64
BONS_NAME_BYTES  = 24  # Entry name field; longer names are stored truncated.

# Header struct (little-endian):
#  0: u32 magic          4: u32 version       8: u32 n_layer      12: u32 n_embd
# 16: u32 n_head        20: u32 n_kv_head    24: u32 head_dim     28: u32 ffn_dim
# 32: u32 vocab         36: u32 rope_theta   40: f32 rms_eps      44: u32 max_ctx
# 48: u32 n_channels    52: u32 n_tensors    56: u32 table_offset 60: u32 payload_offset
# 64: u64 payload_beats (per channel)        72: u32 mode (0=host-owned, 1=fw-owned)
# 76: u32 table_crc32   80: 32B tokenizer sha256                 112: pad to 4096.
_HEADER_FMT = "<IIIIIIIIIIfIIIIIQII32s"

MODE_HOST_OWNED = 0
MODE_FW_OWNED   = 1

def pack_header(cfg, n_channels, n_tensors, payload_beats, table_crc32, tokenizer_sha256=b""):
    table_offset   = BONS_HEADER_SIZE
    payload_offset = BONS_HEADER_SIZE + ceil_div(n_tensors*BONS_ENTRY_SIZE, BONS_HEADER_SIZE)*BONS_HEADER_SIZE
    hdr = struct.pack(_HEADER_FMT,
        BONS_MAGIC, BONS_VERSION,
        cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.ffn_dim, cfg.vocab,
        int(cfg.rope_theta), cfg.rms_eps, cfg.max_ctx,
        n_channels, n_tensors, table_offset, payload_offset, payload_beats,
        MODE_HOST_OWNED, table_crc32, tokenizer_sha256[:32].ljust(32, b"\x00"))
    return hdr.ljust(BONS_HEADER_SIZE, b"\x00")

def unpack_header(data):
    fields = struct.unpack_from(_HEADER_FMT, data, 0)
    keys = ("magic", "version", "n_layer", "n_embd", "n_head", "n_kv_head", "head_dim",
            "ffn_dim", "vocab", "rope_theta", "rms_eps", "max_ctx", "n_channels", "n_tensors",
            "table_offset", "payload_offset", "payload_beats", "mode", "table_crc32",
            "tokenizer_sha256")
    hdr = dict(zip(keys, fields))
    assert hdr["magic"] == BONS_MAGIC, "not a BONS image"
    assert hdr["version"] == BONS_VERSION
    return hdr

# Tensor table entry (64 B):
#  0: u8 role    1: u8 kind    2: u8 flags    3: u8 pad    4: u16 layer   6: u16 group
#  8: u32 rows  12: u32 cols  16: u64 chan_base_beats     24: u64 chan_len_beats
# 32: u32 crc32 (payload, channel-major)     36: u8 bias (B)  37: u8 rshift (R)
# 38: 2B pad   40: 24B ascii name (debug).
_ENTRY_FMT = "<BBBBHHIIQQIBB2x24s"

def pack_entry(role, kind, flags, layer, rows, cols, chan_base_beats, chan_len_beats, crc32,
               bias=0, rshift=0, name=""):
    group = KIND_GROUP.get(kind, 0)
    return struct.pack(_ENTRY_FMT, role, kind, flags, 0, layer, group, rows, cols,
                       chan_base_beats, chan_len_beats, crc32, bias, rshift, name.encode()[:BONS_NAME_BYTES])

def unpack_entry(data, offset=0):
    role, kind, flags, _, layer, group, rows, cols, base, length, crc, bias, rshift, name = \
        struct.unpack_from(_ENTRY_FMT, data, offset)
    return {
        "role": role, "kind": kind, "flags": flags, "layer": layer, "group": group,
        "rows": rows, "cols": cols, "chan_base_beats": base, "chan_len_beats": length,
        "crc32": crc, "bias": bias, "rshift": rshift,
        "name": name.rstrip(b"\x00").decode(errors="replace"),
    }

def crc32(data):
    return zlib.crc32(bytes(data)) & 0xFFFFFFFF

# HBM placement ------------------------------------------------------------------------------------

# Weight payloads start at this byte offset inside every pseudo-channel (per-channel address
# space). The BONS header+table are uploaded verbatim at HBM_WEIGHT_BASE in channel 0 for
# firmware discovery (payloads follow per the header's payload_offset). Channel 0 also hosts the
# CPU's main_ram at channel offset 0, so the CPU heap is capped to this base. 64 MB keeps
# 192 MB/channel for weights: Bonsai-8B (~2.9 GB with embeddings/LM head) fits from N=16 up
# (~180 MB/channel).
# The HBM IP has its AXI switch enabled (USER_SWITCH_ENABLE_00/01), so the *address* selects
# the pseudo-channel and the port it arrives on does not. Channels therefore need distinct
# address windows: with every channel using one base, all N payloads land on top of each other
# and every channel streams identical bytes. Confirmed on hardware 2026-08-03 - `peek <c>
# 4000000` returned byte-identical data for all eight channels, each of which had been uploaded
# a different 58.4 MB payload.
#
# Channel c owns [HBM_WEIGHT_BASE + c*HBM_CHAN_STRIDE, +HBM_CHAN_STRIDE). Constraints:
#  - clear of the CPU's main_ram window, which maps SoC 0x4000_0000 straight through to AXI
#    0x4000_0000 on port 0 (LiteX does not strip the origin for a single mapped channel);
#  - within one stack: the engine's ports are 1..N, all on stack 0, which is the low 4 GB;
#  - a power of two, so the engine's STORE path can shift rather than multiply.
# 2 GB base + 256 MB stride puts N=8 in [2 GB, 4 GB) and leaves the CPU window alone. N=16
# would need 4 GB of windows and so a different split - revisit with the 8B model.
# The base is chosen so each engine channel's window lands on its OWN port's pseudo-channel:
# engine channel c is HBM AXI port c+1 (port 0 = CPU main_ram = PC 0), so its window must be
# PC c+1 = [(c+1)*256 MB, ...). With the old 2 GB base every engine read went to PC 8..15 -
# a far pseudo-channel across the AXI switch's segment links for every single burst. On the
# card that showed up as ~27-34 cycles/beat single-port reads and 4.6 cycles/beat aggregate
# (the segment links saturating), with a read round trip in the microseconds; the same RTL
# against the idealized model streams at 1-2 cycles/beat. Port-matched windows keep every
# read inside its own switch segment.
# 64 MB into PC 1, not PC 1 exactly: the CPU's main_ram bridge passes addresses through
# 1:1, so the CPU lives in PC 4 (0x4000_0000..) - engine channel 3 is port 4, and its
# window necessarily overlaps PC 4. The +64 MB skip keeps that overlap above everything the
# CPU touches at the bottom of its RAM (firmware, header, low heap - all < 8 MB; the
# CPU-side KV cache is placed in a HIGH heap region above the weights tail, see infer.c).
# Payload + KV is ~77 MB < 192 MB, so every channel's data still sits entirely inside its
# own port's pseudo-channel and no read crosses a switch segment.
HBM_WEIGHT_BASE = 0x1400_0000 # PC(c+1) + 64 MB per channel c (port-matched, CPU-safe).
HBM_CHAN_STRIDE = 0x1000_0000 # 256 MB per channel, matching the pseudo-channel size.

# The BONS header+table ("meta") is uploaded by the host to ENGINE channel 0 at this offset
# (the host's PCIe window covers CSRs only, not HBM channel 0 / main_ram); the firmware's
# `loadhdr` command copies it from there to BONSAI_HDR_ADDR via FETCH + memcpy.
HBM_META_OFFSET = HBM_WEIGHT_BASE - 0x0010_0000 # 1 MB before the payloads.

# Where the firmware keeps its CPU-side copy of the BONS header+table, as an offset into
# main_ram. This is a CPU address and has nothing to do with the HBM weight base - deriving it
# from HBM_WEIGHT_BASE put it at 0xC000_0000 once the weight windows moved to 2 GB, i.e. far
# outside main_ram (0x4000_0000, 256 MB). The firmware also places its heap 1 MB above this.
HDR_CPU_OFFSET = 0x0040_0000 # main_ram + 4 MB.
