# TriXium units reference

The hardware blocks of the TriXium core, their descriptor formats and their numerical
contracts. The code is the specification; this page is its map:

| what | file |
| --- | --- |
| descriptor sequencer (format, interlocks) | [`gateware/core/sequencer.py`](../gateware/core/sequencer.py) |
| engine, channel pipe | [`gateware/core/engine.py`](../gateware/core/engine.py), [`gateware/core/pipe.py`](../gateware/core/pipe.py) |
| vector unit, its ISA and golden interpreter | [`gateware/core/vector.py`](../gateware/core/vector.py), [`model/asm.py`](../model/asm.py) |
| scratchpad units | [`gateware/units/`](../gateware/units/) |
| engine numerics (bit-exact spec) | [`model/bitexact.py`](../model/bitexact.py) |
| descriptor encoders, C and Python | [`software/lib/trixium_sequence.h`](../software/lib/trixium_sequence.h), [`model/activation.py`](../model/activation.py), [`model/hadamard.py`](../model/hadamard.py) |
| firmware drivers | [`firmware/seq.h`](../firmware/seq.h), [`firmware/engine.h`](../firmware/engine.h), [`firmware/resident.h`](../firmware/resident.h) |

The C encoders and their Python twins produce identical words (`test/firmware/test_sequence_lib.py`).
They check field encoding only; the hardware checks spans, overlaps and list protection and
reports a failure in the unit's error CSR.

## Scratchpad

256 KB of on-chip RAM, 256 bits wide (one *beat* = 32 bytes = 8 int32/FP32), addressed by
13-bit beat indices; descriptors and encoders use byte addresses (24-bit, 32-byte aligned).
It has exactly two physical ports (more defeat block-RAM inference):

- **R port**: engine x-buffer fill and STORE reads, vector-unit loads, scratchpad-unit reads,
  sequencer descriptor fetches.
- **W port**: engine gather writes (through an 8-deep FIFO), vector-unit stores,
  scratchpad-unit writes.

The CPU sees it uncached at `0xa000_0000` (`SCRATCHPAD_BASE`): writes are posted through a
small FIFO drained when the W port is free, reads borrow a free R-port cycle.

## Sequencer descriptors

A list is `count` 64-byte descriptors (16 little-endian words) at a 32-byte-aligned byte
address `base`; `base + 64*count` must fit the scratchpad. The CPU writes `base`, `count`,
`start` and polls `status` (`busy`, `dcount` = descriptors completed). A descriptor with unit
0 ends the list early. The firmware keeps its lists at `SP_DESC = 0x3D000` (up to 192
descriptors, to 0x40000).

Patches posted by the CPU before `start` land before the first fetch (`sbmode` 1: drained
once at start; 0: every fetch waits on posted writes). A running list's own descriptors are
never patched.

### Word 0

| bits | meaning |
| --- | --- |
| [3:0] | unit: 0 stop, 1 engine op, 2 vector-unit program, 3/4/5/8/11/12/13 scratchpad units |
| [4] | (vec) seed S7 from the engine absmax instead of the previous program's res6 |
| [5] | (engine) capture `k(absmax)` into `kreg[w0[7:6]]` when the op completes |
| [7:6] | (engine) capture register |
| [8] | (engine) also accumulate that k into kreg3 (`kreg3 += k`) |
| [9] | engine: **NOWAIT** (dispatch and move on). Units 3..13: **OVERLAP** (dispatch while a NOWAIT engine op is in flight) |
| [10] | **WAIT-ENGINE** barrier: do not dispatch until the in-flight NOWAIT op completes |
| [11] | (vec) kreg terms are added instead of subtracted |
| [12] | (vec) A5 from a 32-entry scratchpad table: `A5 = LUT[w7 + kreg[sel]]`, table beat index in w10[12:0], `sel` = arg 5's kreg-add select; the index clamps to 0..31 |
| [31:16] | (vec) 2-bit source per argument `i` at `[16+2i +: 2]` |

Vector argument sources: 0 immediate, 1 `k(engine absmax) = max(0, bitlen(absmax) - 14)`,
2 `rpre(prev res6) = bitlen(res6) + 5`, 3 `k(prev res6) = max(0, bitlen(res6) - 14)`.
`w1[31:16]` holds a 2-bit kreg-add select per argument (0 none, n: `arg -= kreg[n-1]`, or `+=`
with w0[11]); kreg3 is reachable by capture only.

### Engine descriptor (unit 1)

| word | bits | field |
| --- | --- | --- |
| w1 | [2:0] | kind (below) |
| | [8] | `ilv`: position-major gather, slot = `row*n_channels + chan` |
| | [9] | `supw`: no scratchpad result writes (argmax tap only, the LM head) |
| | [10] | `rshift_rpre`: rshift = `rpre(prev res6)` instead of w7 |
| | [11] | `qn`: dual-Q GQA (Q8 scores and AXPY; unused by the firmware, 0) |
| | [12] | `xred`: AXPY sums the per-channel partials in hardware, one `cols`-long result |
| w2 | [15:0], [20:16] | `rows` per channel, `rem` (channels `< rem` stream one extra row) |
| w3 | [13:0] | `cols` (K) |
| w4 | [31:0] | `wbase`: per-channel byte address (beat-aligned) |
| w5 | [23:0] | `wlen`: beats per channel |
| w6 | [3:0], [15:8], [31:16] | `sbeats` (scale beats/row), `cbeats` (code beats/row), `qstride` (head-B offset, words; 0) |
| w7 | [6:0], [21:16], [27:22] | bias `B`, result shift `R`, `rshift2` (head-B AXPY pre-shift; 0) |
| w8 | [23:0] | `xsrc`: scratchpad byte address of x (32-byte aligned) |
| w9 | [23:0] | `dst`: scratchpad byte address of the results (4-byte aligned) |
| w10 | [4:0] | `chan`: FETCH source / STORE destination channel |
| w11..w13 | [4:0], [23:0], [7:0] | STORE `rep` packets, `dstride` memory beats, `sstride` scratchpad beats |

### Vector descriptor (unit 2)

`w1[15:0]` = program counter, `w2..w9` = arguments A0..A7 (after source/kreg resolution).

### Interlocks

With NOWAIT the list continues while the engine streams; the list author asserts
independence and the hardware still enforces:

- a second engine descriptor waits until the engine is idle;
- a vec descriptor whose arguments read engine state (KABS source, w0[4], a kreg-add on the
  register the in-flight op will capture) waits for the op to complete;
- units 3..13 wait for the in-flight op unless w0[9] (OVERLAP) is set;
- w0[10] covers dependences the hardware cannot see (softmax reading a scores buffer the
  engine is still filling);
- the stop descriptor and the end of the list drain the in-flight op.

The k capture is applied at done + 2 cycles (the absmax tracker lags the write stage by two).

### Unit errors and ABIs

Each scratchpad unit has `*_abi` (major << 16 | minor) and a sticky `*_error` CSR: bit0 list
bounds/alignment, bit1 malformed descriptor (a reserved bit or word not zero), bits 8..15 the
unit's own error byte (bit 14 = write into the descriptor list). A failure stays until the
engine cluster soft reset (`engine_recover()`); every partial output is invalid.

| unit | ABI | constant |
| ---: | --- | --- |
| 3 Hadamard | 1.0 | `TRIXIUM_HAS_HADAMARD`, `TRIXIUM_HADAMARD_BLOCK` = 1024 |
| 4/5 activation quantize | 1.1 | `TRIXIUM_HAS_ACTIVATION_QUANTIZE` |
| 8 projection restore | 1.0 | `TRIXIUM_HAS_PROJECTION_RESTORE` |
| 11 SiLU | 1.3 | `TRIXIUM_HAS_SILU` |
| 12 norm | 1.2 | `TRIXIUM_HAS_NORM` |
| 13 conv | 1.0 | `TRIXIUM_HAS_CONV` |

The XCU1525 target enables every unit and both stream ops; the FK33 target builds the
engine, sequencer and vector unit only.

## Engine (unit 1)

One AXI reader and one channel pipe per memory channel (16 on both boards). An op fills
every pipe's x-buffer from `xsrc`, gives every reader the same `{wbase, wlen}` in its
channel's window, and the gather writes results to `dst + 4*(chan*rows + row)` (`ilv`:
`row*n_channels + chan`). Rows are sharded per channel; the converter lays the weights out
([`model/layout.py`](../model/layout.py)).

| kind | op | notes |
| ---: | --- | --- |
| 0 | ternary matvec | 128 lanes of {-x, 0, +x} per beat, fp16 scale per 128 |
| 1 | Q8 matvec | 32 lanes of `w8*x8` per beat, fp16 scale per 32 (K-cache scores, Q8 tensors) |
| 2 | FETCH | memory row of channel `chan` -> scratchpad |
| 3 | STORE | scratchpad -> channel `chan`, `rep` packets |
| 4 | AXPY | weighted-V accumulate, `cols` results per channel (`xred`: summed) |
| 5 | DeltaNet stream | `TRIXIUM_HAS_DNSTREAM`, below |
| 6 | attention stream | `TRIXIUM_HAS_ATTNSTREAM`, below |

**Numerics** (`bitexact.matvec_ternary` / `matvec_q8`): per group,
`gs = sum(w*x8)` exact; the fp16 scale is decomposed exactly as `m11 * 2^e16`;
`acc += (gs*m11) << (e16 + B)` in a 48-bit accumulator; `result = acc >> R` (arithmetic) as
int32. The converter picks `B` (largest scale shifted to bit 12) and the smallest `R` that
fits int32, so neither saturates by construction. The semantic scale of a result is
`2^(ex - B + R)`. The engine also tracks the absmax of the gathered results (for `k`) and a
greedy argmax (value, global row; ties to the smallest index).

**Rate.** A pipe takes one 128-trit beat per clock, scale beats included; trit5 weights
(`TRIXIUM_HAS_TRIT5`: 128 trits in 26 of a 32-byte memory beat, `Trit5Unpacker` in front of
the pipe) therefore need only 26/32 of a memory beat per clock, and with `mvbw` reporting the
pipe at 0.96 beats per clock on the YPCB the stream is bound by the pipes, not the memory.
`TriXiumCoreConfig.with_axpy = False` leaves the AXPY datapath (kind 4) out of a build whose
firmware never issues it (the hybrid models): a pipe shrinks from 17.1k to 9.0k LUTs.

**Pipe clock island** (`pipe_clk`, [`gateware/core/island.py`](../gateware/core/island.py)):
the unpacker and the pipe of every channel run in their own clock domain, derived by the SoC
from a PLL (`--pipe-clk-freq`) and reset with the engine (`peng`). Memory beats enter and
results leave through async FIFOs, `start` crosses in a pulse synchronizer, the op parameters
are quasi-static (written at the op's accept, read from `start` on), and the x-buffer's write
port stays in the engine's domain (a dual-clock memory, filled before the first beat). At
1.25x the engine clock a trit5 stream asks the memory for 1.02 beats per engine clock, the
port's rate: the limit moves back to the memory. The unpacker must be inside: in the engine's
domain it emits one pipe beat per engine clock and caps the island at the old rate.

**Readers in the memory's domain** (`readers_cd`,
[`gateware/core/reader_island.py`](../gateware/core/reader_island.py), `--ddr-readers` on the
YPCB target and the simulator): with the pipes in their island the next cap is the channel's
256-bit AXI at the engine clock (3.2 GB/s at 100 MHz against a DDR3-1066 controller's ~3.7 GB/s
share per channel; the stream measured 0.95 memory beats per engine clock). The reader then
runs in the controller's own clock (`reng`: the `ddr` clock with the engine's reset) on a
read-only AXI port that reaches the bridge in that domain with no crossing (the write half of
the port, the loader's, keeps its sys-side path: `AXIResetShim(reads=, writes=)`,
`AXICrossing(channels=)`, `engine_ddr_port(read_axi=)`). The engine commands it through a small
async FIFO (every channel ready, then one command each: a FIFO's `ready` is room, not an idle
reader), the weight beats go straight into the pipe island's input FIFO (`PipeIsland(in_cd=)`),
and only the FETCH / DeltaNet / attention beats cross back to the engine through an async FIFO,
steered by two quasi-static levels (the op's kind, the prefetch gate) crossed by MultiReg. The
soft reset holds `reng` with the engine; the read shims' hold and busy cross by MultiReg and
`busy` covers the readers' reset release. The trit5 pipe then caps the stream at ~1.03 memory
beats per engine clock at a 125 MHz pipe (26 trits of 32 per beat). At 133 MHz the stream reaches
1.04 beats per engine clock on the YPCB-00338, the DDR3-933 controller's own rate after refresh
and row switches; a 143 MHz pipe adds nothing there. At DDR3-1066 with the pipes at 143 MHz the
stream reaches 1.11.

### Op 5: DeltaNet stream

One op runs every head of a 27B DeltaNet layer
([`gateware/units/deltanet_stream.py`](../gateware/units/deltanet_stream.py)). The recurrent
state never enters the scratchpad: lane `l` (= channel `l`, 16 lanes) owns rows
`l, l+16, ...` of every 128x128 head and streams them from its channel's reader and back to
its channel's writer, in place. Fields: `rows` = heads, `xsrc` = the head record table,
`wbase`/`wlen` = every channel's state span, `B` bit 0 = treat the incoming state as +0 (the
state after reset). Head record (one beat): `[q, k, v, out byte addresses, decay, beta,
scale, 0]`; records may alias (heads sharing a key head share q/k). Per row, rounded after
every FP32 operation:

    m = m*decay; d = d + m*k[i]            (pass 1, i in order, d from +0)
    delta = (v - d)*beta
    m = m + delta*k[i]; o = o + m*q[i]     (pass 2, i in order, o from +0)
    out = o*scale

Pass 1 of head h+1 overlaps pass 2 of head h: `cols*rows/lanes` = 1,024 clocks per head.

### Op 6: attention stream

One lane per query head (24 lanes on the XCU1525, 4 kv groups, head dim <= 256, T <= 1024;
[`gateware/units/attention_stream.py`](../gateware/units/attention_stream.py)). K or V rows are
position-striped across channels (`rows = T // n_ch`, `rem = T % n_ch`, `cbeats` beats per
row) and merged back into position order. `xsrc` points at a one-beat record:

| word | field |
| --- | --- |
| w0 | `mode` (0 scores, 1 values) \| `nq << 8` \| `n_rep << 16` \| `nkv << 24` |
| w1 | T (1..t_max) |
| w2 | hd (power of two, 8..8*hb_max) |
| w3, w4 | source byte address (q rows / p rows), lane stride |
| w5, w6 | destination byte address (scores / outputs), lane stride |
| w7 | scale (FP32, mode 0) |

Mode 0: `s[l][t] = (sum_i q[l][i]*k[t][i], i in order from +0) * scale`, written as beats of
8 positions (padding positions hold junk). Mode 1: `o[l][i] = sum_t p[l][t]*v[t][i]`, t in
order from +0. The softmax between them stays on the CPU, so both passes are bit-exact with
the firmware loops.

## Vector unit (unit 2)

A microcoded 16-lane SIMD engine for the remaining elementwise and reduction work (BFP
norms, softmax, K-segment `reduce_acc`, the Qwen3 path's activations). Single-issue, in
order, one instruction per cycle, zero-bubble hardware loops.

- 16 vector registers x 16 lanes x int32 (`TRIXIUM_VEC_VREGS`), 8 scalar registers (48-bit),
  8 argument registers A0..A7.
- Q15 LUT ops (EXP2F, RSQRT, RECIP) with linear interpolation; tables from `bitexact.py`.
- Instruction RAM: `IMEM_DEPTH` = 1024 words. The firmware uploads every program at init;
  `asm.emit_c()` asserts they fit (an overflow would wrap silently onto program 0).
- The interpreter in `asm.py` is the golden model; the RTL matches it bit for bit
  (`test/gateware/core/test_vector.py`).

## Unit 3: integer Hadamard

Rotates complete 1,024-element int8 blocks into raw signed int32 vectors (natural Sylvester
order, no normalization, no rounding) and writes one uint32 raw absmax per block
([`gateware/units/hadamard.py`](../gateware/units/hadamard.py)).

| word | field |
| --- | --- |
| w1 | src: int8 blocks |
| w2 | signs: packed bits, low bit first, 1 = negate (128 B per block) |
| w3 | dst: int32, `4096*blocks` bytes |
| w4 | blocks [15:0], >= 1 |
| w5 | stats: one uint32 absmax per block |
| w6 | inverse [0]: 0 = `H(sign*x)`, 1 = `sign*(H*x)` |

Rate (32 lanes, 1,024-element blocks): the block's four sign beats, then one data read per
clock into the kernel (32), ten butterfly stages of 16 clocks + 1 drain, 128 int32 output
beats: ~347 clocks per block, of which the output is write-port bound.

The output and stats spans must not overlap each other or the inputs. Error byte: bit0
config, bit1 bounds, bit2 overlap, bit3 core. The image's dense rotations are
`H*diag(sign)/32`: `resident_init()` recovers the signs from row 0 of each rotation tensor,
and the `1/32` is the `-5` in unit 5's exponent rule.

## Units 4 and 5: activation quantize / requantize

[`gateware/units/activation.py`](../gateware/units/activation.py).

**Unit 4** (FP32 -> Q8, per block): `w1` src, `w2` dst, `w3` exponents (one int32 per
block), `w4[15:0]` elements **per block** (multiple of 32), `w5[15:0]` blocks. Per block:
scan the absmax, `e8 = frexp_exponent(absmax) - 7` (0 for a zero block), then
`q = clamp(round_ties_away(x * 2^-e8), -127, 127)` (the firmware's `lroundf`, not RNE).
NaN/Inf input, or a nonzero absmax whose e8 would leave [-127, 121], fails.
With several blocks of at most 1,024 elements the unit runs ping-pong: the scan copies the
block into one half of a 2 x 4 KB buffer, and the convert reads it from there while the next
block's scan reads the scratchpad - one scratchpad pass per block, ~137 clocks per 1,024
elements instead of ~262. A nonfinite input in the next block then fails the command before
the current block retires (partial output is invalid either way).

**Unit 5** (raw Hadamard int32 -> one whole-vector Q8): `w1` raw int32 src, `w2` dst, `w3`
per-block input e8, `w4` per-block raw absmax (unit 3's stats), `w5` result exponent (a
reserved beat), `w6[15:0]` blocks of 1,024. The result exponent is
`E = max over nonzero blocks of (e_in + bitlen(absmax) - 12)`, and each element is quantized
ties-away with exponent `E - e_in + 5`. Metadata are validated first (`e_in` in [-127, 121],
absmax <= 131072, every raw value consistent with its declared absmax).

Error byte: bit0 geometry, 1 capacity, 2 overlap, 3 non-finite input, 4 unsafe exponent,
5 internal, 6 protected list, 7 invalid rotation metadata.

## Unit 8: projection restore

int32 -> FP32 ([`gateware/units/restore.py`](../gateware/units/restore.py)): `w1` src, `w2`
dst, `w3` exponent beat (signed e8 in lane 0), `w4[15:0]` elements (1..65535), `w5` signed
adjustment (normally the matrix's `R - B`). `out = RNE_f32(x) * 2^(e8 + adj)`; the sum is
checked against [-149, 127] before use; an overflow to infinity fails. `src == dst` is
allowed. The unit tracks the greedy argmax (first max, NaN frozen) of everything it writes
since list start: `restore_argmax_idx` / `_val`, the LM head's token.

Error byte: bit0 geometry, 1 bounds, 2 overlap, 3 unsupported exponent, 4 FP32 overflow,
5 internal, 6 protected output.

## Unit 11: SiLU-multiply

`dst[i] = m[i] * siluf(g[i])`, bit-exact with the firmware's table lookup
([`gateware/units/silu.py`](../gateware/units/silu.py), arithmetic in
[`model/silu_hw.py`](../model/silu_hw.py)). Four kernels, one element per
clock each; two resident table pairs.

| word | bits | field |
| --- | --- | --- |
| w1 | | g (FP32); LOAD: the table image |
| w2 | | m (FP32) |
| w3 | | dst: `== w1` (in place) or disjoint |
| w4 | | fixup bitmap, `ceil(elements/32)` words, all written |
| w5 | [15:0] | elements |
| w6 | [0] | op: 0 RUN, 1 LOAD |
| | [1] | `unit_m`: m = 1.0 (w2 unused) |
| | [2] | `table_only`: every element outside the table range is a fixup (any table, e.g. the sigmoid) |
| | [3] | `table_set`: which resident pair LOAD writes and RUN reads |
| w7 | [15:0], [23:16], [31:24] | `seg_beats`, `perm_a`, `perm_b`: output segment `s` goes to `(s % a)*b + s/a` |

Table: `lut[0..4096]` over [-16, 16] at 128 points per unit and `dlut[i] = lut[i+1] - lut[i]`;
the LOAD image is 1,025 beats (lut in beats 0..512, dlut from beat 513). Per element:
`a = RNE(g + 16) * 128`, `i = floor(a)`, `s = fma(frac, dlut[i], lut[i])`, `r = m * s` (RNE).
**Fixups**: g in the tails (-80 <= g <= -16, g == 16), non-finite g or m, subnormal m, or a
product that is not normal. The unit writes g to that lane and sets its bitmap bit; the CPU
recomputes those elements with its own code. `silu_fixups` counts them over a list.

The DeltaNet gated norm uses the permutation to write the head outputs in grouped order;
the attention output gate uses table set 1 (the sigmoid) with `table_only`.

## Unit 12: RMSNorm / L2 norm (+ RoPE)

For `nseg` segments of `n` FP32 values, exact IEEE binary32
([`gateware/units/norm.py`](../gateware/units/norm.py)):

    RMS (op 0): ss = sum64(x*x); r = 1/sqrt(ss/n + eps);       dst[i] = (x[i]*r)*w[i]
    L2  (op 1): ss = sum64(x*x); inv = 1/max(sqrt(ss), eps);   dst[i] = (x[i]*inv)*1.0

| word | bits | field |
| --- | --- | --- |
| w1, w2, w3 | | src, dst (`== src` or disjoint), weight (RMS, n values, read once) |
| w4 | [15:0], [31:16] | n (8..8192, multiple of 8), nseg (>= 1) |
| w5 | | eps (FP32 bits) |
| w6 | [0] | op: 0 RMS, 1 L2 |
| | [12:8] | `rope_hb`: neox RoPE on the first `16*rope_hb` values of each segment (<= 256), 0 = none |
| | [28:16] | `rope_tab`: beat index of the table, `rope_hb` cos beats then `rope_hb` sin beats |
| w7 | [15:0], [31:16] | src / dst stride in beats between segments (0 = contiguous, else >= n/8) |

No fixups. Two segments overlap in the pipeline (pass 1 and the tree of segment i+1 under the
scalar chain and pass 2 of segment i).

## Unit 13: causal conv / add

[`gateware/units/conv.py`](../gateware/units/conv.py), exact binary32, no fixups.

| word | field |
| --- | --- |
| w1, w2, w3 | x0, x1, x2: the window slots, oldest first |
| w4 | x3: the new input |
| w5 | taps: 4 FP32 per channel, interleaved as in the model |
| w6 | dst (may overwrite x3) |
| w7 | channels [15:0] (multiple of 8) \| op [16] |

Op 0: `y[c] = (((+0 + w[4c]*x0[c]) + w[4c+1]*x1[c]) + w[4c+2]*x2[c]) + w[4c+3]*x3[c]`.
Op 1 (residual add, `bns_seq_desc_add`): `y[c] = x0[c] + x3[c]`, dst == x0 or x3. The SiLU
after the conv is unit 11 with `unit_m`.

## Unit 14: attention softmax

[`gateware/units/softmax.py`](../gateware/units/softmax.py), exact binary32, no fixups.

| word | field |
| --- | --- |
| w1 | src: the first row of scores (32-byte aligned; in place) |
| w2 | n [15:0] positions per row (1..4096) \| rows [31:16] (1..256) |
| w3 | row stride in beats (0 = ceil(n/8)) |

For every row, hybrid.c's `softmax_rows` bit for bit: `mx = max(s[t])` from -1e30 (a NaN
never wins), `e[t] = softmax_exp(s[t] - mx)` (the canonical exp below), `sum = (((+0 + e[0])
+ e[1]) + ...)`, `inv = 1/sum`, `p[t] = e[t]*inv`. Lanes past n in a row's last beat are not
written. Three passes: the maximum 8 lanes per clock; the exp one element per clock through a
pipeline of exact operators (`FP32Add`, `FP32Mul`, the fused `FP32Fma` for the six polynomial
stages), eight rows interleaved so each row's sequential sum gets one addition every 8 clocks
(past the adder's latency) and stays in the CPU's order; one reciprocal per row on the
sequential divider, then the scale 8 lanes per clock. 16 heads at T = 256: ~6,200 clocks
(the CPU loop: ~166,000). With the unit the attention's scores, softmax and weighted-V
passes are one list (`hybrid.c`, `usm`): no main-RAM round trip, no CPU between them; the
console `softmaxunit off` keeps the CPU path for a same-boot comparison.

## Canonical reductions

Every FP32 reduction the firmware, the Python mirror (`model/mirror.py`) and the
units share follows one order, chosen so a unit with an ordinary pipelined adder reproduces
it at one beat per clock. The firmware is built with `-ffp-contract=off`.

**Sum of squares (`sum64`)**, `sumsq64` in `firmware/hybrid.c`, `_sumsq64` in the mirror,
all RMSNorm and L2 norms:

1. 64 partials: `P[k] = (((+0 + t_k) + t_{k+64}) + t_{k+128}) + ...`, `t_i = RNE(v_i*v_i)`,
   every add RNE, in increasing index order; an empty partial stays +0.
2. Halving tree: for `w = 32, 16, 8, 4, 2, 1`: `P[j] = P[j] + P[j+w]` for `j < w`.
3. Result `P[0]`. Element `8*beat + lane` goes to partial `8*(beat % 8) + lane`: each partial is
   revisited every 8 clocks.

**Reciprocal.** The scalar that follows is one correctly rounded operation, then one multiply
per element: `r = 1/sqrt(ss/n + eps)` (RMS), `inv = 1/max(sqrt(ss), eps)` (L2) - never a divide
per element.

**Softmax exp.** `trixium_softmax_exp(x)` (`firmware/fastmath.h`): the bounded exp
(range reduction to `[-ln2/2, ln2/2]`, degree-6 polynomial in FMAs, exact `2^n`) on
[-80, 0], +0 below -80, a NaN argument returned. Arguments are `s - max`, so the maximum's
term is exactly 1. The DeltaNet decay `exp(g)` uses the same function (`trixium_decay_exp`).
The probabilities are the exps times one reciprocal of their sum per head (`1.0f/sum` once,
then multiplies: ggml's order too; before 2026-10-18 a divide per position, the CPU's `fdiv`
at every element of every head at every token - the long-context pole).
The mirror's rv32f profile reproduces it (`softmax_exp_rv32`, qualified against the compiled
firmware under QEMU in `test/model/test_rv32_math.py`).

**Sigmoid.** `trixium_sigmoid_lookup`: the SiLU lookup's form on a sigmoid table (built by
`trixium_sigmoid_point` at the 4,097 grid points), tails from the same bounded exp. Unit 11
runs it with `table_only`, the CPU patches the fixups.

**Sequential orders kept.** The attention dot products and weighted-V sums (op 6 reproduces
the sequential order, one lane per head), the softmax denominator (CPU, sequential) and the
DeltaNet recurrence (op 5, in order).

## Overlap and scratchpad arbitration

The two scratchpad ports are granted per cycle, so the engine can stream while the vector
unit or a scratchpad unit works on disjoint regions:

- **R port**: vector-unit / scratchpad-unit read > sequencer descriptor fetch > engine. A
  vector load claims the port for exactly its address cycle; the engine's fill/STORE address
  holds on those cycles.
- **W port**: engine gather writes go through an 8-deep FIFO and never stall the gather (its
  merge and result counting must not stall). Vector/unit stores take the port while the FIFO
  holds fewer than 6 entries; at the mark the engine drains and the store holds, re-driving
  the same data (numerics cannot change). The engine never waits on the vector unit, so the
  FIFO always drains; `done` waits for it to empty. FETCH beats are back-pressured by the
  same mark.
- Scratchpad units are mutually exclusive and share the vector unit's client slot
  (`ScratchpadArbiter`, whose base client is the vector unit when the bitstream has one: the
  hybrid-only builds leave it out, the arbiter then serves the units alone); a unit runs under
  an engine op only with w0[9] (OVERLAP).
- CPU sideband reads/writes take free cycles below all of them.

Firmware rules (`engine_submit` / `engine_join` in `firmware/engine.c`, and NOWAIT lists):
only independent pairs overlap - disjoint scratchpad footprints, no absmax/res6/kreg chaining
across the pair, no engine absmax/argmax read and no second engine op before the join, and no
CPU scratchpad loops in the window (the sideband stalls while the engine is busy). The
counters `cyc_eng`, `cyc_vec`, `cyc_both` measure the overlap (`trixium_smi`).
`test/gateware/core/test_scratchpad.py` checks concurrent runs against serialized ones.

## DeltaNet recurrence arithmetic

The 27B's Gated DeltaNet recurrence runs in exact FP32 on engine op 5 (above), bit-exact with
the firmware's CPU loop; the state lives in engine DDR as 128x128 FP32 per head, in the
lane-striped layout only op 5 reads and writes (`state_to_lanes` / `lanes_to_state` in
`deltanet_stream.py`).

Heads are processed key-head major (the `rep` value heads of one key head consecutive, so
their records share q and k). The model's output is tiled `[hd, nk, rep]` (value head `h`
uses key head `h % nk`); the converter folds the out-projection rotation in grouped order
(`[hd, rep, nk]`), so the firmware writes the gated-norm output in grouped order directly
(unit 11's output permutation, `perm_a = nk`, `perm_b = rep`) and the golden permutes before
the rotation.

The mirror's `fast=True` profile (`hybrid27_generate`) models a quantized alternative - q/k/v
as int8 per head, the state as int16 with one ratcheting exponent per head, Q15 decay, exact
int32 dots, a two-pass vector-unit recurrence - for precision studies; the hardware does not
implement it.

## Scratchpad maps

**27B hybrid, resident lists** (`firmware/resident.c`, `hybrid.c`):

| region | offset | use |
| --- | --- | --- |
| `RS_Q8B` | 0x00000 | per-block Q8, unit 4 out / unit 3 in (transient, <= 17,408 B) |
| `RS_Y` | 0x04800 | int32 engine results, 96 KB (<= 24,576 rows per chunk) |
| `RS_F` | 0x1C800 | FP32 source, then raw rotation, then FP32 result (to 0x32000) |
| `RS_XW` | 0x32000 | whole-vector Q8 (persists from a prepare to its matvecs) |
| `RS_E8` | 0x36400 | its exponent beat; `RS_EXP` 0x36440, `RS_MAX` 0x364C0 per-block metadata |
| `XRES_SP` | 0x36600 | the resident residual stream |
| `RS_SIGN` | 0x3C000 | Hadamard sign bits, 128 B per block, every width (persists) |
| `SP_DESC` | 0x3D000 | descriptor lists |

Per layer, transient regions reuse `RS_Y`/`RS_F`: DeltaNet `DNR_QKV` = 0x04800; attention
`AT_Q`/`AT_KST`/`AT_VST`/`AT_REC`/`AT_O` inside `RS_Y`, scores/p, gated outputs, fixup maps
and the unit-11 table image inside `RS_F`.

A projection is two lists:

```text
prepare (once per input vector):
  [1] FETCH src (FP32, main RAM) -> RS_F
  [4] quantize RS_F -> RS_Q8B, one e8 per 1,024-block
  [3] FWHT RS_Q8B, signs -> RS_F (raw int32) + per-block absmax
  [5] requantize RS_F -> RS_XW (whole-vector Q8) + RS_E8
      (unrotated inputs: [4] alone, one block, straight to RS_XW / RS_E8)
matvec (per projection):
  [1] engine matvec per K-segment, xsrc = RS_XW (+ offset), dst = RS_Y
  [2] reduce_acc: wrapping int32 sum of the K-segment partials
  [8] restore RS_Y -> RS_F with e8 + (R - B)
  [1] STORE RS_F -> main RAM
```

**Qwen3 (1.7B / 8B) path** (`firmware/infer_internal.h`): `SP_X` 0x00000 (int8 matvec input),
`SP_Y` 0x04000 / `SP_YU` 0x10000 (int32 results), `SP_H` 0x1C000, `SP_XR` 0x1E000 (residual),
`SP_G` 0x20000, `SP_U` 0x26000, `SP_D` 0x2C000 (int16 activations), `SP_FET` 0x30000,
`SP_KVP` 0x34000, `SP_SC` 0x36000 (scores), `SP_Q` 0x38000, `SP_K` 0x3A000, `SP_V` 0x3B000,
`SP_TAB` 0x3B800, `SP_FET2` 0x3C000, `SP_DESC` 0x3D000.
