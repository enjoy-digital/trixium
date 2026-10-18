# Related work and acknowledgements

TriXium is a thin layer of new ideas on top of a lot of other people's work. This page says
whose, and how TriXium compares with the projects closest to it.

## Built on

- **[LiteX](https://github.com/enjoy-digital/litex)** (Enjoy-Digital and contributors): the SoC
  builder, the CPU integration, the BIOS and `liblitedram`'s calibration code, the AXI/wishbone
  interconnect, the build flow. Everything that is not the inference core is LiteX.
- **[LiteDRAM](https://github.com/enjoy-digital/litedram)**: the DDR3/DDR4 controllers and PHYs
  (USPDDRPHY on the XCU1525, the K7DDRPHY with CK phase shift on the YPCB-00338), the AXI
  frontends, the BIST.
- **[LitePCIe](https://github.com/enjoy-digital/litepcie)**: the PCIe endpoints, the DMA and the
  generated Linux driver behind `/dev/trixium0`.
- **[LiteX-Boards](https://github.com/litex-hub/litex-boards)**: the platform files of the three
  cards; the YPCB-00338-1P1's (`ypcb_00338_1p1`) follows TiferKing's documentation of the board.
- **[Migen](https://github.com/m-labs/migen)** (M-Labs): the HDL every gateware block is written in.
- **[VexiiRiscv](https://github.com/SpinalHDL/VexiiRiscv)** (SpinalHDL): the RISC-V CPU that runs
  the firmware (its predecessor VexRiscv ran the first tokens).
- **[Verilator](https://www.veripool.org/verilator/)**: the whole-SoC simulation and the compiled
  unit tests; most of the design was proven there before it met a card.
- **OpenOCD** and **openFPGALoader**: board programming over JTAG.
- **QEMU**: an independent oracle for the firmware's float paths (the RV32F `expf` agreed bit for
  bit on a 20,481-point sweep).
- **AMD/Xilinx Vivado** (2024.1) builds the bitstreams, and the FK33's HBM2 controller is AMD's
  hard-block IP: the two parts of the stack that are not open source (there is no open
  alternative for the hard HBM stack).

## Models and formats

- **[PrismML](https://prismml.com/news/ternary-bonsai)**: the natively ternary **Bonsai** models
  TriXium runs (Ternary-Bonsai 1.7B and 8B, Ternary-Bonsai-2 27B,
  [Hugging Face](https://huggingface.co/prism-ml)), the `PQ2_0` / `PTQ1_0` weight formats, and
  their llama.cpp fork, which TriXium's golden model reproduces exactly and which served as the
  reference for the 27B's tokens.
- **Qwen** (the Qwen team, Alibaba): the Qwen3 and Qwen3.5 architectures the Bonsai models are
  trained on, including the Qwen3.5 hybrid's Gated DeltaNet and gated-attention layers.
- **Gated DeltaNet** (gated delta-rule linear attention): the recurrent layers of the 27B, which
  got their own streaming unit (engine op 5).
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** / ggml / **GGUF**: the model container the
  converter reads, codes copied bit for bit.
- **[BitNet](https://github.com/microsoft/BitNet)** (Microsoft): the 1.58-bit (ternary) LLM line of
  work that made "multiply by -1, 0 or +1" a serious idea.

## Ideas borrowed, and comparison points

- **[openTPU](https://github.com/FeSens/openTPU)** (FeSens): an open int8/4-bit systolic
  inference accelerator with an ISA and a compiler, on the same YPCB-00338 card. The closest
  comparison, the reference for "how busy can the DRAM be", and the source of the DDR3 bring-up
  technique on the card's HR banks (fine CK phase scan, per-bit read framing, a traffic test at
  every step), reimplemented in LiteX/LiteDRAM. It also runs on LiteDRAM.
- **TeLLMe** (ternary BitNet on a Kria KV260): the fair comparison that kept us honest in August.
- **TerEffic**, **FlightLLM**, **Grasshopper** (Zettascale) and **Redwood Nano**: published FPGA
  LLM accelerators, compared below.
- **RFNoC** (Ettus Research): the counter-example behind "specialise at build time" (a
  composable block framework spends fabric on its interconnect; TriXium fixes the topology at
  build time, [`architecture.md`](architecture.md#specialised-at-build-time)).

## People and machines

- **SQRL** and the crypto-mining community, for the FK33 and XCU1525 cards (now second-hand);
  **Inspur**, for the YPCB-00338.
- **Claude** (Anthropic): the agent that wrote the code, guided by Enjoy-Digital; **OpenAI Codex**
  sessions contributed external reviews and some of the September work ([`method.md`](method.md)).

[![Limits removed, upstream: what TriXium needed from LiteX and LiteDRAM went back into them (CK phase shift PHY and leveling, registered write path, reset-less data path, pipelined reads, MPR read leveling, HBM2 IP generation, faster Verilog generation), with the PR numbers](images/video-upstream.jpg)](images/video-upstream.jpg)

What went back: the fixes TriXium needed are upstream or on their way (DDR4 MPR read leveling,
HBM2 IP generation, pipelined up-converter reads, faster Verilog generation, the RMW write
path, the DFI-level ECC in progress).

## openTPU on the same card

The YPCB-00338-1P1 (Kintex-7 XC7K480T, two DDR3-1066 channels, 17.1 GB/s) runs both designs.
openTPU's numbers are from its README at commit 2026-10-14 (build B at 133.33 MHz, decode on the
device after a 512-token prompt; ours after a 3-token prompt):

| | openTPU, 4-bit weights, int8 head | TriXium, ternary |
| --- | --- | --- |
| ~2B | Qwen3.5-2B 12.09, Gemma 4 E2B 12.14 (4-bit head), LFM2-2.6B 10.96 tok/s | Bonsai 1.7B **35.2 tok/s**; Bonsai-2 2B (ternary Qwen3.5-2B) **14.99 tok/s**, hybrid build at DDR3-933 |
| 3-4B | SmolLM3-3B 8.74, Phi-4-mini 6.56, Qwen3.5-4B 5.88 tok/s | Bonsai-2 4B (ternary Qwen3.5-4B) **6.88 tok/s**, same build |
| 8B | (its images stop at 4 GiB) | Bonsai 8B **8.11 tok/s** |
| DRAM while decoding | 91-94 % of the 17.1 GB/s peak | 78 % (1.7B), 79 % (8B); ~89-91 % while the engine streams |
| DRAM bytes read per token (~2B model) | ~1.3 GB (Qwen3.5-2B, 4-bit) | ~0.38 GB (Bonsai 1.7B): ~3.5x less |
| largest model in the card's 4 GB | ~4B (its images stop at 4 GiB) | 8B (2.85 GiB of weights) |
| memory controller | LiteDRAM, DDR3-1066 (133.33 MHz) | LiteDRAM, DDR3-1066 (133.33 MHz) |
| clocks | one clock, 133.33 MHz | core 125 MHz, DDR controllers 133.33 MHz |

The Bonsai-2 2B / 4B (2026-10-18) are ternary versions of the same Qwen3.5-2B / 4B openTPU runs at
4 bits, and they decode faster (14.99 against 12.1, 6.88 against 5.9 tok/s) with ~2.5x fewer
bytes per token: the hybrid build uses about 40% of the DRAM bandwidth (the 2B: ~0.41 GB per
token at 15.0 tok/s, at DDR3-933 on this card), the rest being per-token work (the scratchpad
units and the CPU on each layer's serial chain) not yet overlapped with the weight streams. That is where the ternary
format's advantage is still to be collected: a small speed win so far, a large bandwidth one.

Bytes per token come from each design's measured speed and DRAM use: 17.1 GB/s x DRAM busy /
tok/s. Two small open projects, one person each (give or take an AI agent), opposite trade-offs.
Per parameter, 2.5-2.8x openTPU's decode rate, all of it from the bytes per weight (1.73 bits
against ~4.5): openTPU keeps the DRAM busier, TriXium moves fewer bytes. Different models, so a
speed comparison, not a quality one. The philosophies differ: openTPU moved flexibility into
software (an ISA and a compiler, any off-the-shelf model), TriXium into the weight format
(natively ternary models) with a fixed engine. openTPU also runs a single faster clock (133.33
MHz against our 125 MHz core), so the per-parameter gap is the format, not the clock. Our gap is the serial work of a layer
(attention, residual tails, the SiLU step, CPU glue: ~25 % of a 1.7B token), see
[`performance-next-steps.md`](performance-next-steps.md).

## How TriXium compares

Batch-1 decode is bandwidth-bound on every system below, so the comparable quantities are bytes
per parameter, DRAM saturation and exposed non-streaming work, not raw tok/s or clock. Numbers
for other systems are as published.

| system | platform | model | tok/s | note |
| --- | --- | --- | ---: | --- |
| TriXium | FK33 (VU33P, HBM2), 212.5 MHz | Ternary-Bonsai 1.7B / 8B | 106.3 / 30.3 | |
| TriXium | XCU1525 (VU9P, 4x DDR4-1500) | Ternary-Bonsai-2 27B | 4.03 | |
| TriXium | YPCB-00338 (K7-480T, 2x DDR3-1066) | Ternary-Bonsai 1.7B / 8B | 35.2 / 8.11 | |
| openTPU (FeSens) | Kintex-7 XC7K480T, 2x DDR3-1066 (17.1 GB/s), 133.33 MHz | Qwen3-0.6B int8 / Qwen3.5-4B 4-bit | 21.6 / 5.9 | int8 systolic unit, ISA + compiler, one bitstream for many models |
| Grasshopper (Zettascale) | VU47P build, 225 MHz | Gemma 3 1B int8 | 51.0 | 89 % of peak memory bandwidth, ~21 W core |
| TeLLMe v2 | Kria KV260, 17.1 GB/s, 4.8 W | BitNet 0.73B (W1.58A8) | 25 | ternary transformer with attention |
| FlightLLM | Alveo U280, 460 GB/s | LLaMA2-7B ~INT4 | 55 | |
| TerEffic | Alveo U280 | 2.7B matmul-free (HGRN) | 727 | no attention, no KV cache |
| Redwood Nano | Versal VPK180 | Qwen3-0.6B INT8 | 12.1 | |

What the comparisons taught:

- **TeLLMe v2 was the fair comparison in August, and it was ahead.** Same W1.58A8 scheme, a real
  transformer, ~1.8x our size-normalised throughput at 5.874 tok/s with a third of our
  bandwidth. Our MAC array was idle 92 % of the token: the gap was control, not silicon. Closing
  it is the 2 -> 101 tok/s story.
- **TerEffic's 727 tok/s is a different workload**: a ternary matmul-free RNN with no attention
  and no KV cache, on >700k LUTs and a >$10k card. Its ternary core cannot compute
  activation x activation attention scores.
- **openTPU and Grasshopper saturate memory better**: 82-94 % DRAM utilization for openTPU, 89 %
  for Grasshopper, against ~60 % token average for TriXium's weight streams on the XCU1525
  (v22-era: 22.6 GB/s for the 27B, 92 % during streaming) and 43 % for the FK33 1.7B. TriXium
  leads on absolute throughput and params x tok/s (FK33: 172 Gparams/s on the 1.7B, 240 on the
  8B; openTPU's best dense ~28; Grasshopper 51) almost entirely through ternary density and
  provisioned channels.
- **Both point the same way for the next step**: the control plane off the critical path
  ([`architecture.md`](architecture.md#why-a-risc-v--microcode--streams-and-not-a-tpu)).
