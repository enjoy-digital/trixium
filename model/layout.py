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
2. BONS image format: the on-disk/in-memory weight image (header + tensor table + payloads).
3. Tensor kinds, flags and roles, and the memory placement constants.

Geometry: one HBM pseudo-channel AXI beat is 256 bits = 32 bytes. For ternary tensors
(group=128), one beat carries exactly one group's 2-bit codes (128 codes x 2b = 32B), so one
beat feeds the 128-lane adder tree for one cycle. Row packet = all group scales (fp16, packed
16/beat) followed by all code beats, rows back-to-back:

  row r: [scale beat 0 .. scale beat S-1][code beat 0 .. code beat G-1]   (S = ceil(G/16))

KIND_TERNARY5 stores the same trits 5 per byte (base 3, 26 bytes per 128-group, lane 5i+k is
digit k of byte i), the row padded to a whole beat: S + ceil(26G/32) beats in memory, still G
code beats for the engine's pipe (the Trit5Unpacker rebuilds the 2-bit beats in front of it).

Tensors are block-sharded by output row across N channels (channel c gets rows
[c*M/N, (c+1)*M/N)), with the same relative addresses in every channel, so a single
{base, len} descriptor drives all N weight streamers.
"""

import math
import zlib
import struct

# Geometry -----------------------------------------------------------------------------------------

BEAT_BYTES      = 32            # One HBM AXI beat: 256 bits.
GROUP_TERNARY   = 128           # Weights per ternary group (one fp16 scale, one code beat each).
GROUP_Q8        = 32            # Weights per int8 group (one fp16 scale, one code beat each).
SCALES_PER_BEAT = BEAT_BYTES//2 # 16 fp16 scales per beat.

# Tensor Kinds -------------------------------------------------------------------------------------

KIND_TERNARY = 0 # 2-bit codes {0,1,2}, group=128, w = (q-1)*scale.
KIND_Q8      = 1 # int8 quants, group=32, w = q*scale.
KIND_F16     = 2 # Raw fp16 rows (no scales).
KIND_F32     = 3 # Raw fp32 rows (no scales; norms etc, consumed by CPU/vector unit).
KIND_TERNARY5 = 5 # KIND_TERNARY's codes stored 5 trits per byte (with_trit5 engines).
KIND_NORM16  = 4 # int16 rows; for the ".w16" norm companions the entry's bias byte is the
                 # tensor's exponent (int8): w = q * 2^bias. Rope/embedding NORM16 rows keep
                 # their own fixed formats and bias 0.

# NORM16 exponent, per tensor: the smallest exponent whose 15-bit mantissa still holds the
# tensor's absmax. A fixed exponent does not fit: Qwen3-1.7B's norm gammas reach 153, while the
# tiny models' gammas are all small (so their gates cannot see a clipped gamma).
NORM16_EXP_MAX_FALLBACK = -13 # For an all-zero tensor.

def norm16_exp(values):
    amax = float(abs(values).max()) if values.size else 0.0
    if amax == 0.0:
        return NORM16_EXP_MAX_FALLBACK
    return math.ceil(math.log2(amax/32767.0))

KIND_NAMES = {
    KIND_TERNARY : "ternary",
    KIND_Q8      : "q8",
    KIND_F16     : "f16",
    KIND_F32     : "f32",
    KIND_NORM16  : "norm16",
    KIND_TERNARY5: "ternary5",
}

KIND_GROUP = {
    KIND_TERNARY : GROUP_TERNARY,
    KIND_Q8      : GROUP_Q8,
    KIND_TERNARY5: GROUP_TERNARY,
}

TRIT5_GROUP_BYTES = 26 # ceil(128/5).

def is_ternary(kind):
    """Ternary codes, 2-bit (KIND_TERNARY) or 5 trits per byte (KIND_TERNARY5)."""
    return kind in (KIND_TERNARY, KIND_TERNARY5)

# Tensor flags.
FLAG_REPLICATED = 0x01 # Tensor payload identical in every channel (not row-sharded).
FLAG_KSEG       = 0x02 # One K-segment of a matvec too wide for one engine op: the entry's
                       # `kseg` (byte 3) is the segment index, siblings share role/layer/kind
                       # and the tensor's bias/rshift; the firmware sums their int32 results
                       # (the 27B's FFN down projection: K = 17408 > the op's 14-bit cols and
                       # the 16 KB int8 input buffer).
KSEG_MAX        = 16384  # The widest K one engine op takes.
KSEG_MAX_Q8     = 255*32 # Q8: one code beat per 32 columns, 8-bit cbeats.

# Tensor Roles -------------------------------------------------------------------------------------

(ROLE_TOK_EMB, ROLE_OUT_NORM, ROLE_LM_HEAD,
 ROLE_ATTN_NORM, ROLE_ATTN_Q, ROLE_ATTN_K, ROLE_ATTN_V, ROLE_ATTN_OUT,
 ROLE_ATTN_Q_NORM, ROLE_ATTN_K_NORM,
 ROLE_FFN_NORM, ROLE_FFN_GATE, ROLE_FFN_UP, ROLE_FFN_DOWN,
 ROLE_ROPE_TAB,
 ROLE_TOK_EMB16, ROLE_TOK_EMB16_EXP) = range(17)

# ROLE_TOK_EMB16: the token embedding rows pre-dequantized to the firmware's int16 BFP form
# (bit-exact with embed_row_bfp / the sim's embed_bfp_int), one NORM16 row per token, so a
# forward starts with a 128-beat engine fetch instead of ~47k cycles of CPU decode and
# uncached scratchpad stores. ROLE_TOK_EMB16_EXP: the per-row exponent, 16 int16 per beat,
# replicated.

# The 27B's hybrid layers (BONS v3). Gated DeltaNet: the fused [q|k|v] projection, the z gate
# projection, the out projection (its input in grouped head order), the beta and alpha head
# projections (Q8), the conv taps (F32, kernel rows), A = -exp(A_log) and the dt bias (F32,
# one row of value heads), the per-head norm (F32, one row of head dim). The attention layers'
# q projection carries [q | gate] per head when the header says attn_gated. Hadamard: the
# block matrices R diag(s) per input width as ternary tensors (rows = width, cols = block),
# layer = the width's index in the header's list.
(ROLE_ATTN_QKV, ROLE_ATTN_ZGATE, ROLE_SSM_OUT, ROLE_SSM_BETA, ROLE_SSM_ALPHA,
 ROLE_SSM_CONV, ROLE_SSM_A, ROLE_SSM_DT, ROLE_SSM_NORM,
 ROLE_HADAMARD) = range(17, 27)
# Roles 27-30 are reserved (unused).

def hadamard_layer(width_index, block_index):
    """The entry `layer` of a ROLE_HADAMARD block tensor (u16: width index in the header's
    list, then the block)."""
    assert 0 <= width_index < 4 and 0 <= block_index < 256
    return width_index*256 + block_index

# The ternary roles the engine streams that KIND_TERNARY5 stores packed (the token embedding stays
# 2-bit: the firmware decodes its rows on the CPU when the int16 table is absent).
TRIT5_ROLES = (ROLE_ATTN_Q, ROLE_ATTN_K, ROLE_ATTN_V, ROLE_ATTN_OUT,
               ROLE_FFN_GATE, ROLE_FFN_UP, ROLE_FFN_DOWN, ROLE_LM_HEAD,
               ROLE_ATTN_QKV, ROLE_ATTN_ZGATE, ROLE_SSM_OUT)

ROLE_NAMES = {
    ROLE_TOK_EMB       : "tok_emb",
    ROLE_OUT_NORM      : "out_norm",
    ROLE_LM_HEAD       : "lm_head",
    ROLE_ATTN_NORM     : "attn_norm",
    ROLE_ATTN_Q        : "attn_q",
    ROLE_ATTN_K        : "attn_k",
    ROLE_ATTN_V        : "attn_v",
    ROLE_ATTN_OUT      : "attn_out",
    ROLE_ATTN_Q_NORM   : "attn_q_norm",
    ROLE_ATTN_K_NORM   : "attn_k_norm",
    ROLE_FFN_NORM      : "ffn_norm",
    ROLE_FFN_GATE      : "ffn_gate",
    ROLE_FFN_UP        : "ffn_up",
    ROLE_FFN_DOWN      : "ffn_down",
    ROLE_ROPE_TAB      : "rope_tab",
    ROLE_TOK_EMB16     : "tok_emb16",
    ROLE_TOK_EMB16_EXP : "tok_emb16_exp",
    ROLE_ATTN_QKV      : "attn_qkv",
    ROLE_ATTN_ZGATE    : "attn_zgate",
    ROLE_SSM_OUT       : "ssm_out",
    ROLE_SSM_BETA      : "ssm_beta",
    ROLE_SSM_ALPHA     : "ssm_alpha",
    ROLE_SSM_CONV      : "ssm_conv",
    ROLE_SSM_A         : "ssm_a",
    ROLE_SSM_DT        : "ssm_dt",
    ROLE_SSM_NORM      : "ssm_norm",
    ROLE_HADAMARD      : "hadamard",
}

# Row Packet Math ----------------------------------------------------------------------------------

def ceil_div(a, b):
    return (a + b - 1)//b

def deltanet_state_bytes_per_channel(cfg, n_channels):
    """FP32 recurrent-matrix storage needed by the hardware DeltaNet path.

    Whole value-head matrices are striped over the engine channels. Return the largest
    channel's allocation so the converter's fit check covers a non-even final stripe too.
    The short convolution history remains CPU-resident and is not part of this region.
    """
    if not cfg.n_dn_layers:
        return 0
    heads      = cfg.n_dn_layers*cfg.dn_num_v_heads
    head_bytes = cfg.dn_v_head_dim*cfg.dn_k_head_dim*4
    return ceil_div(heads, n_channels)*head_bytes

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
    """Code beats per row as the engine's pipe counts them (one 128-group per ternary beat)."""
    if is_ternary(kind):
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

def mem_code_beats(cols, kind):
    """Code beats per row in memory (KIND_TERNARY5: 26 bytes per group, the row padded)."""
    if kind == KIND_TERNARY5:
        return ceil_div(TRIT5_GROUP_BYTES*(cols//GROUP_TERNARY), BEAT_BYTES)
    return code_beats(cols, kind)

def row_beats(cols, kind):
    """Beats per row in memory."""
    return scale_beats(cols, kind) + mem_code_beats(cols, kind)

def shard_rows(rows, n_channels, channel):
    """Row range [start, count) of `channel` for a block-sharded tensor."""
    assert rows % n_channels == 0, f"rows {rows} not divisible by {n_channels} channels"
    per = rows//n_channels
    return channel*per, per

# BONS Image Format --------------------------------------------------------------------------------
#
# Single file: [header 4 KiB][tensor table, 64 B/entry, padded to 4 KiB][channel 0 payload]
# [channel 1 payload]...[channel N-1 payload]. All channel payloads have the same length
# (payload_beats). Each channel's payload is uploaded at the same base offset of its window;
# the header+table go to channel 0's staging area (the MB below its window) and the firmware's
# `hdrload` copies them to main_ram, so it walks the model without host-side sideband.

BONS_MAGIC       = 0x534E4F42 # "BONS" little-endian.
BONS_VERSION     = 3  # 2: NORM16 .w16 entries carry their exponent in bias.
                      # 3: the hybrid fields at 112; a v2 image reads as v3 with every one of
                      #    them zero (a plain Qwen3 model).
BONS_VERSIONS_OK = (2, 3)
BONS_HEADER_SIZE = 4096
BONS_ENTRY_SIZE  = 64
BONS_NAME_BYTES  = 24  # Entry name field; longer names are stored truncated.

# Header struct (little-endian):
#  0: u32 magic          4: u32 version       8: u32 n_layer      12: u32 n_embd
# 16: u32 n_head        20: u32 n_kv_head    24: u32 head_dim     28: u32 ffn_dim
# 32: u32 vocab         36: u32 rope_theta   40: f32 rms_eps      44: u32 max_ctx
# 48: u32 n_channels    52: u32 n_tensors    56: u32 table_offset 60: u32 payload_offset
# 64: u64 payload_beats (per channel)        72: u32 mode (0=host-owned, 1=fw-owned)
# 76: u32 table_crc32   80: 32B tokenizer sha256
# v3 (all zero on a Qwen3 model):
# 112: 2 x u64 layer_types  (bit l set = layer l is Gated DeltaNet; 128 layers)
# 128: u16 dn_v_heads 130: u16 dn_k_heads 132: u16 dn_k_dim 134: u16 dn_v_dim 136: u16 dn_conv
# 138: u16 rope_dims (0 = whole head)     140: u16 hadamard_block (0 = plain basis)
# 142: u8 attn_gated  143: u8 hyb_flags (bit 0: the emb16 rows are un-rotated)
# 144: 4 x u32 hadamard_widths (the ROLE_HADAMARD tensors, by layer index)
# 160: u8 reserved (0)                     161: pad to 4096.
_HEADER_FMT        = "<IIIIIIIIIIfIIIIIQII32s"
_HEADER_V3_FMT     = "<QQHHHHHHHBBIIII"
_HEADER_V3_OFF     = 112
HYB_EMB16_RESTORED = 0x01

MODE_HOST_OWNED = 0
MODE_FW_OWNED   = 1

def pack_header(cfg, n_channels, n_tensors, payload_beats, table_crc32, tokenizer_sha256=b""):
    table_offset   = BONS_HEADER_SIZE
    table_size     = ceil_div(n_tensors*BONS_ENTRY_SIZE, BONS_HEADER_SIZE)*BONS_HEADER_SIZE
    payload_offset = BONS_HEADER_SIZE + table_size
    hdr = struct.pack(_HEADER_FMT,
        BONS_MAGIC, BONS_VERSION,
        cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.ffn_dim, cfg.vocab,
        int(cfg.rope_theta), cfg.rms_eps, cfg.max_ctx,
        n_channels, n_tensors, table_offset, payload_offset, payload_beats,
        MODE_HOST_OWNED, table_crc32, tokenizer_sha256[:32].ljust(32, b"\x00"))
    hdr = hdr.ljust(_HEADER_V3_OFF, b"\x00")
    lt = 0
    for l in range(cfg.n_layer):
        if cfg.layer_type(l) == "deltanet":
            lt |= 1 << l
    widths = sorted(cfg.hadamard_signs) if getattr(cfg, "hadamard_signs", None) else []
    assert len(widths) <= 4
    hdr += struct.pack(_HEADER_V3_FMT,
        lt & ((1 << 64) - 1), lt >> 64,
        cfg.dn_num_v_heads, cfg.dn_num_k_heads, cfg.dn_k_head_dim, cfg.dn_v_head_dim,
        cfg.dn_conv_kernel, cfg.rope_dims, cfg.hadamard_block,
        1 if cfg.attn_gated else 0, HYB_EMB16_RESTORED if cfg.hadamard_block else 0,
        *(widths + [0]*(4 - len(widths))))
    hdr += struct.pack("<B", 0) # Reserved.
    return hdr.ljust(BONS_HEADER_SIZE, b"\x00")

def unpack_header(data):
    fields = struct.unpack_from(_HEADER_FMT, data, 0)
    keys   = ("magic", "version", "n_layer", "n_embd", "n_head", "n_kv_head", "head_dim",
              "ffn_dim", "vocab", "rope_theta", "rms_eps", "max_ctx", "n_channels", "n_tensors",
              "table_offset", "payload_offset", "payload_beats", "mode", "table_crc32",
              "tokenizer_sha256")
    hdr = dict(zip(keys, fields))
    assert hdr["magic"] == BONS_MAGIC, "not a BONS image"
    assert hdr["version"] in BONS_VERSIONS_OK, hdr["version"]
    # A v2 image reads as v3 with every hybrid field zero.
    v3 = (0,)*15
    if hdr["version"] >= 3:
        v3 = struct.unpack_from(_HEADER_V3_FMT, data, _HEADER_V3_OFF)
    lt = v3[0] | (v3[1] << 64)
    hdr["layer_types"] = tuple("deltanet" if (lt >> l) & 1 else "attn" for l in range(hdr["n_layer"]))
    hdr["dn_num_v_heads"], hdr["dn_num_k_heads"], hdr["dn_k_head_dim"], hdr["dn_v_head_dim"], \
        hdr["dn_conv_kernel"], hdr["rope_dims"], hdr["hadamard_block"] = v3[2:9]
    hdr["attn_gated"], hdr["hyb_flags"] = v3[9], v3[10]
    hdr["hadamard_widths"] = tuple(w for w in v3[11:15] if w)
    return hdr

# Tensor table entry (64 B):
#  0: u8 role    1: u8 kind    2: u8 flags    3: u8 kseg   4: u16 layer   6: u16 group
#  8: u32 rows  12: u32 cols  16: u64 chan_base_beats     24: u64 chan_len_beats
# 32: u32 crc32 (payload, channel-major)     36: u8 bias (B)  37: u8 rshift (R)
# 38: 2B pad   40: 24B ascii name (debug).
_ENTRY_FMT = "<BBBBHHIIQQIBB2x24s"

def pack_entry(role, kind, flags, layer, rows, cols, chan_base_beats, chan_len_beats, crc32,
               bias=0, rshift=0, name="", kseg=0):
    group = KIND_GROUP.get(kind, 0)
    return struct.pack(_ENTRY_FMT, role, kind, flags, kseg, layer, group, rows, cols,
                       chan_base_beats, chan_len_beats, crc32, bias, rshift,
                       name.encode()[:BONS_NAME_BYTES])

def unpack_entry(data, offset=0):
    role, kind, flags, kseg, layer, group, rows, cols, base, length, crc, bias, rshift, name = \
        struct.unpack_from(_ENTRY_FMT, data, offset)
    return {
        "role": role, "kind": kind, "flags": flags, "kseg": kseg, "layer": layer, "group": group,
        "rows": rows, "cols": cols, "chan_base_beats": base, "chan_len_beats": length,
        "crc32": crc, "bias": bias, "rshift": rshift,
        "name": name.rstrip(b"\x00").decode(errors="replace"),
    }

def crc32(data):
    return zlib.crc32(bytes(data)) & 0xFFFFFFFF

# HBM Placement ------------------------------------------------------------------------------------

# Engine channel c owns [HBM_WEIGHT_BASE + c*HBM_CHAN_STRIDE, +HBM_CHAN_STRIDE) and its weight
# payload starts at the bottom of that window. The HBM IP has its AXI switch enabled, so the
# *address* selects the pseudo-channel, not the port: the channels need distinct windows
# (with one shared base all N payloads would land on top of each other). Constraints:
#  - Port-matched: engine channel c is HBM AXI port c+1 (port 0 = CPU main_ram), so its window
#    sits in PC c+1 and no read crosses an AXI switch segment (a far pseudo-channel costs the
#    segment links on every burst).
#  - 64 MB into PC c+1: the CPU's main_ram bridge passes addresses through 1:1, so the CPU
#    lives in PC 4 (0x4000_0000..), which engine channel 3's window overlaps. The skip keeps
#    that overlap above everything the CPU touches at the bottom of its RAM (firmware, header,
#    low heap; the CPU-side KV cache sits in a high heap region, see infer.c).
#  - Within stack 0 for the engine ports; with N=16, channel 15's window is stack 1's first
#    pseudo-channel, still its own port's (the write path carries 33-bit addresses with the
#    channel offset added in hardware, like the read path).
#  - A power-of-two stride, so the engine's STORE path can shift rather than multiply.
# This leaves 192 MB/channel for weights + KV.
HBM_WEIGHT_BASE = 0x1400_0000 # PC(c+1) + 64 MB per channel c (port-matched, CPU-safe).
HBM_CHAN_STRIDE = 0x1000_0000 # 256 MB per channel, matching the pseudo-channel size.


def ddr_chan_stride(ctrl_size, chan_mod, weight_base=HBM_WEIGHT_BASE):
    """Window stride for chan_mod windows on one DDR controller: the controller above
    weight_base split evenly (no power of two needed when the hardware knows chan_mod: the
    STORE/upload paths add a constant per window). A 64 KB multiple, so the bank skew alone
    sets the windows' bank offsets. Each window is the stride minus the skew's
    (chan_mod - 1)*chan_skew (the last one ends at the controller's end)."""
    return ((ctrl_size - weight_base)//chan_mod) & ~0xFFFF

# Where the firmware keeps its CPU-side copy of the BONS header+table, as an offset into
# main_ram. This is a CPU address, unrelated to the HBM weight base (deriving it from
# HBM_WEIGHT_BASE can put it outside main_ram). The firmware places its heap 1 MB above.
HDR_CPU_OFFSET = 0x0040_0000 # main_ram + 4 MB.
