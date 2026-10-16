# TriXium architecture

TriXium runs PrismML's Bonsai ternary language models (Ternary-Bonsai 1.7B / 8B, Qwen3
topology; Ternary-Bonsai-2 27B, Qwen3.5 hybrid with Gated DeltaNet layers) entirely on an FPGA
card, designed bit-exact against a Python reference (exact in simulation; for the open
card-level issues see [boards: known issues](boards.md#known-issues)). This document explains
how a token is computed, what each block does, and why the design is a small RISC-V CPU with
a microcoded vector unit and streaming fixed-function units rather than a TPU-style systolic
array.

## The problem: batch-1 decode is a memory stream

Generating one token runs every layer once, and every weight is read once per token. At
batch 1 there is no weight reuse: the arithmetic is a sequence of matrix-*vector* products
and the token time is, to first order,

    t_token ≈ weight bytes / sustained memory bandwidth + per-token work that is not hidden

So the design question is not "how many MACs" but "how close to the memory's peak can the
weights stream, and how much of everything else can run underneath that stream".

Ternary weights change the arithmetic as well as the bandwidth:

![A ternary dot product: select +x, skip, or -x, then add](images/video-ternary.jpg)

- **Bytes.** A Bonsai matrix weight is a trit {-1, 0, +1}, stored as 2 bits plus one fp16
  scale per 128 weights: 2.125 bits per weight (1.73 with the YPCB's 5-trits-per-byte codes),
  ~4x fewer bytes than int8 and ~2x fewer than 4-bit formats. At equal bandwidth that is the
  throughput ratio.
- **Arithmetic.** Multiplying by a trit is selecting +x, 0 or -x: a group of 128 weights is an
  adder tree (cheap LUT logic, sized to match the bytes the memory delivers), then one
  multiply of the group's sum by its fp16 scale. The non-matrix work (norms, attention,
  SiLU) is ordinary arithmetic, and the 27B's small DeltaNet beta/alpha projections are int8
  (the engine's Q8 path).

## The system

![TriXium SoC](images/architecture.svg)

**Targets.** A target (`trixium_xcu1525.py`, `trixium_fk33.py`, `trixium_ypcb.py`,
`trixium_sim.py`) supplies the
clocks, the CPU and SoC bus, PCIe and the memory system, and instantiates one `TriXiumCore`
with three standard interfaces: `memory_ports` (one 256-bit AXI master per engine channel, in
the sys domain), `bus` (a 32-bit wishbone slave onto the scratchpad, the CPU's window) and
`upload` (a stream sink fed by the host's PCIe DMA). The XCU1525 maps its 16 channels onto four LiteDRAM DDR4
controllers (in their own clock domain, crossed at the AXI level); the FK33 onto 16 HBM2
pseudo-channels; the YPCB maps eight channels onto two LiteDRAM DDR3 controllers;
the Verilator simulation uses memory models with the same contract.

![FK33: LitePCIe Gen3 x4 and 16 direct HBM2 engine ports](images/board-fk33.svg)

![XCU1525: LitePCIe Gen3 x4 and four LiteDRAM DDR4 controllers, 4 x 4 = 16 engine ports](images/board-xcu1525.svg)

![YPCB-00338: LitePCIe Gen2 x8 and two LiteDRAM DDR3 controllers, 2 x 4 = 8 engine ports](images/board-ypcb.svg)

**Weights.** The converter (`model/convert.py`) turns a GGUF into a BONS image: every
matrix is split row-wise across the channels so each channel streams its own slice in
parallel, with per-channel windows in memory. Matrices wider than one engine op are split
into K-segments whose partial sums are added in the scratchpad.

### How a token flows

1. **The CPU runs the model loop** (`firmware/infer*.c` for Qwen3, `firmware/hybrid.c` for
   the 27B): for each layer it posts work and patches the few data-dependent words of
   pre-built descriptor lists (positions, KV slots).
2. **The sequencer executes the lists** (`gateware/core/sequencer.py`): 64-byte descriptors in
   the scratchpad, dispatched back-to-back to the engine, the vector unit or a scratchpad
   unit, with interlocks so independent work overlaps (an engine op can stream while
   vector/unit descriptors run underneath it). Dispatching from the CPU would cost ~1k
   cycles per op; a list runs a whole projection group with one start and one poll.
3. **The engine streams the weights** (`gateware/core/engine.py`, `gateware/core/pipe.py`): every channel
   pipe reads its slice of the matrix, multiplies it against the activation vector held in
   its x-buffer (128 ternary lanes or 32 Q8 lanes per cycle) and the gather writes int32
   results into the scratchpad. The same pipes stream the 27B's DeltaNet state (op 5) and
   the attention K/V rows (op 6) through dedicated stream units.
4. **The activations stay resident**: a projection's input is quantized, rotated
   (integer Hadamard) and requantized in the scratchpad by the units, its int32 result is
   restored to FP32 there, and the SiLU/norm/conv units apply the layer's nonlinearities -
   the CPU only touches the data between lists when it has to.
5. **The LM head** streams like any projection, with a hardware argmax tap on the results;
   the token goes back to the host (or into the next forward).

Sequencer units (descriptor `w0[3:0]`):

| unit | function | file |
| ---: | --- | --- |
| 1 | engine op: ternary / Q8 matvec, fetch, store, AXPY, DeltaNet stream, attention stream | `gateware/core/engine.py` |
| 2 | vector-unit program (microcode) | `gateware/core/vector.py` |
| 3 | integer Hadamard (FWHT) on 1,024-element blocks | `gateware/units/hadamard.py` |
| 4 | FP32 -> Q8 block quantization | `gateware/units/activation.py` |
| 5 | whole-vector requantization after the rotation | `gateware/units/activation.py` |
| 8 | int32 -> FP32 projection restore (+ argmax) | `gateware/units/restore.py` |
| 11 | SiLU-multiply (exact FP32, table + fixups) | `gateware/units/silu.py` |
| 12 | RMSNorm / L2 norm (+ RoPE) | `gateware/units/norm.py` |
| 13 | DeltaNet causal conv + residual add | `gateware/units/conv.py` |

Descriptor fields, bounds, numerics contracts, the resident lists and the scratchpad map:
[`units.md`](units.md).

### Numerics and the mirror

Every hardware path is bit-exact with a Python mirror of the firmware
(`model/mirror.py`): the int8/int16 block-floating-point
activations, the integer Hadamard, the exact FP32 units (IEEE binary32 with subnormals,
round-to-nearest-even) and the CPU's own float arithmetic. The mirror is the oracle of the
simulation gate (FP32 checkpoints compared at every layer) and of the board checks (the
same checkpoints read back from the card). Exactness is a design constraint, not a test
afterthought: each unit was added only once it reproduced the CPU path bit for bit.

## Why a RISC-V + microcode + streams, and not a TPU

![Why not a TPU](images/why-not-tpu.svg)

A TPU-style accelerator is a systolic array of MACs with a compiler that tiles matrix
*multiplications* onto it. It shines when each weight fetched from memory is used many times:
training, prefill, large batches. For batch-1 decode on a memory-bound card that reuse does
not exist:

- **A systolic array would sit idle.** Each weight is used once per token. The engine needs
  exactly as many lanes as the memory can feed (the XCU1525's 16 channels x 128 ternary
  lanes match ~4 DDR4 controllers' bandwidth); a larger array waits on memory, a 2D array
  has nothing to reuse. TriXium's MAC "array" is adder trees sized to the byte rate.
- **Ternary removes the multipliers** a TPU is built around. Sign-select + adder trees cost
  LUTs, not DSP blocks, and scale with bandwidth.
- **The models are not just matmuls.** The 27B interleaves Gated DeltaNet layers (a
  recurrent state update per head per token), gated attention with partial RoPE, per-head
  norms, SiLU gating, causal convolutions and a KV cache. Each needs a different dataflow,
  and they evolve from one model generation to the next. A fixed matmul engine (TerEffic-
  style ternary cores) cannot run attention or the recurrence at all; a TPU needs a
  compiler that maps each of them, and still leaves them as separate phases.
- **Programmability where it is cheap, hardware where it pays.** The control plane is a
  RISC-V running C (the model loop is ordinary firmware: new topologies are firmware
  changes), the irregular elementwise work started on a microcoded vector unit, and an
  operation moved into a fixed-function unit only when profiling showed it on the critical
  path (SiLU, norms, the DeltaNet stream, attention, the resident quantize/restore chain).
  The hardware/software boundary moved where the measurements said, without redesigns.
- **It is FPGA- and LiteX-native.** Every block is ordinary LiteX/Migen gateware with a
  standard interface; the same core runs on DDR4 and HBM boards and in Verilator, and the
  bit-exact mirror makes each change verifiable before it costs a 6-hour place-and-route.

openTPU (FeSens) is the instructive contrast on the same YPCB-00338 card: an int8/4-bit
systolic machine with an ISA and a compiler, one bitstream for many model families, flexibility
in software where TriXium put it in the weight format; the numbers are in
[`related-work.md`](related-work.md#opentpu-on-the-same-card).

### Specialised at build time

The closest software analogy is a hand-written GPU kernel
rather than a general-purpose accelerator: each bitstream is built for one model family and
one board (channel count, lane widths, weight code, which units exist), so no logic is spent
on flexibility the model never uses - no configurable crossbar between generic blocks, the
way a composable framework like RFNoC spends fabric on its interconnect. The cost is a build
(hours of Vivado) per family; the return is a datapath sized to the memory, on smaller and
cooler FPGAs than a generic design would need.

## What this costs (the trade-offs we accept)

- **Per-token serial work.** Everything that does not stream (embedding, norms, the
  DeltaNet heads, attention scores, list setup, the CPU's glue) adds to the weight stream:
  on the XCU1525's 27B it is ~10.5 M of the 32 M cycles per token. The weights stream at
  ~2.3x the 1-beat-per-channel-per-cycle floor, against 82-94% DRAM utilization for
  openTPU's sequencer-driven design: our control plane is the measured gap.
- **A small CPU in the loop.** The VexiiRiscv runs at the fabric clock; anything left on it
  is slow, and keeping it off the critical path is a constant effort.
- **Firmware complexity.** The hybrid model loop, the resident lists and their overlap
  rules are intricate C; the mirror keeps them honest but it is code to maintain.
- **Batch-1 only.** Prefill and batched decode do not reuse weights in this design: prompts
  are processed token by token.
- **Natively ternary models only.** The density advantage requires models trained ternary
  (PrismML's Bonsai family); arbitrary off-the-shelf models would need a different engine.
- **Area per unit.** Each fixed-function unit costs LUTs and timing closure work on a
  near-full die; the 27B configuration fills the XCU1525's VU9P to its routing limit.

## Measured

The decode rates, cycles per token and clocks of every board and model, with their caveats, are
in [`results.md`](results.md); where the remaining time goes and what is next, in
[`performance-next-steps.md`](performance-next-steps.md).

## Gateware conventions

### Resets

Only control state is reset: valids and handshake flags, FSM states, counters and pointers
that decide what happens next, sticky errors, status and performance counters a CPU reads,
and anything the engine soft reset (`gateware/core/reset.py`) must return to a known value.
Datapath registers carry no reset: operands and results, pipeline payloads and their tags
(qualified by a valid), latched commands and addresses (loaded before the state that reads
them), accumulators cleared by a start or first-beat condition, delay lines. In Migen:
`Signal(..., reset_less=True)` at the declaration, or `reset_less_registers(module,
keep=valids)` (`gateware/core/datapath.py`) for a pure pipeline. When in doubt, keep the reset.

A reset on a datapath register is not free on the FPGA: it is fanout on a high-fanout net
(the `eng` soft reset reaches the whole engine cluster), a set/reset pin that splits control
sets, and it blocks SRL inference for delay lines and register absorption into DSPs and
BRAMs. Beware `Signal.like`, which copies `reset_less` (and the reset value) from its source:
a valid derived from a reset-less signal must say `reset_less=False`.

Measured on the core (XCU1525 configuration, out-of-context synthesis on the VU9P): flip-flops
with a set/reset pin in use 301,038 -> 110,377, LUTs 591,744 -> 577,580, shift-register LUTs
3,462 -> 7,156 (delay lines leave the flip-flops), DSPs 1,330 -> 1,320. Control sets rise
(3,353 -> 5,868: clock-enable-only groups split from the reset ones); post-synthesis slack
stays positive (+1.44 -> +1.08 ns, an estimate before placement).

The simulations cannot see a missing reset on their own (Verilog `reg x = init` starts every
register at its reset value). `./trixium_sim.py --random-init SEED --build` gives every
register no reset reaches a random power-on value (`software/lab/randinit.py`; MultiReg
synchronizer stages excepted: crossings start from INIT by design), and `--gate-pre
"gen 2 5 6 7;wedge"` soft-resets the engine in the middle of an op before the traced
generation. Both must leave the gate bit-identical ([`simulation.md`](simulation.md)).

## Verification

- **Unit and integration tests** (`test/`): Migen and compiled Verilator simulations of every
  module against Python models, the descriptor encoders (C vs Python), the converter and the mirror.
- **Verilator SoC simulation** (`trixium_sim.py`): the same core, firmware and memory
  contract as the boards; `./trixium_sim.py --build` builds it in ~3.5 minutes,
  `--firmware` swaps in a new firmware in ~17 s, `--gate` compares a traced generation with the mirror
  (87 FP32 checkpoints, bit-identical). Details: [`simulation.md`](simulation.md).
- **Boards**: `trixium deploy` checks timing, uploads, boots and compares the generated
  tokens with the reference before measuring speed (the FP32 checkpoints: `software.lab.mirror`).
