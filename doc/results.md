# Results

Batch-1 greedy decode on the card, short context; every deploy checks the first tokens of a
generation against the reference before reporting a number (for what that check does not see,
the [known issues](boards.md#known-issues)). How the numbers evolved: [`timeline.md`](timeline.md).

## On the cards

| board | model | tok/s | cycles/token | clock |
| --- | --- | ---: | ---: | --- |
| XCU1525 (4x DDR4-1500) | Ternary-Bonsai-2 27B | 4.03 | 34.2 M | sys 137.5 MHz (v31b) |
| XCU1525 | Ternary-Bonsai 8B | 14.9 | 8.37 M | sys 125 MHz |
| XCU1525 | Ternary-Bonsai 1.7B | 53.8 | 2.32 M | sys 125 MHz |
| FK33 (HBM2, 16 channels) | Ternary-Bonsai 8B | 30.3 | 7.02 M | sys 212.5 MHz |
| FK33 | Ternary-Bonsai 1.7B | 106.3 | 2.00 M | sys 212.5 MHz |
| YPCB-00338 (2x DDR3-1066) | Ternary-Bonsai 8B | 8.11 | 15.41 M | sys 125 MHz |
| YPCB-00338 | Ternary-Bonsai 1.7B | 35.2 | 3.55 M | sys 125 MHz |
| YPCB-00338 (hybrid build, 2x DDR3-1066, 4 channels, ECC + forward retry) | Ternary-Bonsai-2 4B | 9.23 | 10.8 M | sys 100 MHz, pipes 143 MHz, readers in the DDR clock |
| YPCB-00338 (hybrid build) | Ternary-Bonsai-2 2B | 20.33 | 4.92 M | sys 100 MHz, pipes 143 MHz, readers in the DDR clock; 19.49 tok/s at 256 tokens, 15.81 at 1,024 |
| Efinix Ti375 C529 (LPDDR4x x32, both AXI targets, 4 channels, lite units) | Ternary-Bonsai-2 2B | 14.32 | 6.98 M | everything at 100 MHz (`ti375_v19`); 13.96 tok/s at 256 tokens, 11.97 at 1,024, 10.44 at 1,800 |

The fine print: the XCU1525 runs its DDR4 at 1500 MT/s, past the PHY's qualified rate (an
approved overclock), and the builds close timing to the project's empirical gate
(`software/lab/timing.py`), not to full static sign-off ([`boards.md`](boards.md)). The video's
1.7B climb ends at its peak build (101.9 tok/s at 212.5 MHz); the shipping build reaches 106.3
(reset-less datapath, fused residual tail). Where the remaining time goes:
[`performance-next-steps.md`](performance-next-steps.md).

The Ti375 kit is the first small board: one LPDDR4x x32 behind the hard DDR controller, shared by
the CPU and four engine channels, the lighter unit set (no softmax unit, RoPE on the CPU), the
weights uploaded over UDP, everything at 100 MHz. The first bitstream (v7, 8.86 tok/s) used one of
the controller's two 512-bit AXI targets: 6.4 GB/s, half the LPDDR4x's rate, and its weight
streams ran at ~0.46 beats per channel per clock, 92% of that port - two thirds of a forward
waited on it. Five steps since, each measured on the kit:

| step | 2B tok/s at 64 / 256 / 1,024 | what moved |
| --- | --- | --- |
| v7: one AXI target, DeltaNet stream width 4 | 8.86 / 8.05 / 5.41 | |
| v8: both AXI targets (`--ddr-targets 2`, litex#2688) | 11.94 / 10.46 / - | streams 0.46 -> 0.75 beats per channel per clock (9.6 GB/s); FFN projections 2.93 -> 1.94 M cycles, LM head 2.20 -> 1.35 M |
| v9: + DeltaNet stream width 8 (`--dn-sublanes 2`) | 12.94 / 11.20 / 6.66 | the recurrence heads 1.39 -> 0.76 M |
| v9 + the embedding restore on the Hadamard unit (firmware) | 13.28 / 11.45 / 6.75 | the embedding row 436 k -> 245 k cycles |
| v11: + the channel windows skewed by 8 KB (`--chan-skew 0x2000`) | 13.63 / 11.70 / 6.84 | the channels stop hitting the same DRAM banks: streams 0.75 -> 0.78 beats per channel per clock |
| v15: + the attention softmax unit (`--with-softmax`) | 14.08 / 13.20 / 9.31 | the CPU's softmax (3.9 M cycles per forward at 1,024 tokens) moves to the unit: attention 8.0 -> 4.1 M |
| v18: + the attention stream at 8 lanes / 2 groups (`--attn-lanes 8 --attn-groups 2`) | 14.33 / 13.97 / 10.93 | both kv heads in one command per layer: the values pass at 1,024 tokens 3.25 -> 1.66 M cycles |
| v19: + the attention stream to 2,048 positions (`--attn-t-max 2048`), registered read data in the activation and softmax units | 14.32 / 13.96 / 11.97 | past 1,024 positions v18 ran attention on the CPU (~4.6 s per token; the 1,024 bench's last forwards); 1,800 tokens at 10.44 tok/s; worst slack +0.78 ns (108.5 MHz achievable) |
| v20: + the microSD card (`--with-sdcard`) | 14.34 / 13.98 / 11.99 | the weights and tokenizer from the card at power-up (1-bit on the kit: 436 MB in 74 s); 105.3 MHz achievable, 96.05% XLRs |

A v9 forward at 64 tokens is 7.3 M cycles (`hybprof`): the DeltaNet layers 2.6 M, the FFNs 1.9 M,
the LM head 1.35 M, the attention layers 1.1 M, the embedding row 0.25 M. The image has no int16
embedding table (it would fill the 1 GB), so each token's row is restored from the rotated
ternary table: split exactly into int8 chunks, rotated on the Hadamard unit, recombined and
rounded on the CPU (`embres unit|cpu`). The board's
reference tokens are its image's own (that restore is exact, one LSB away from the converter's
table), checked against the mirror and deterministic run to run ([`boards.md`](boards.md#efinix-ti375-c529)).

The Bonsai 2 2B/4B on the YPCB-00338 are still some way from its memory limit: the 2B streams
~0.41 GB per token, so 20.3 tok/s uses about half of the DDR3-1066 bandwidth (4.8 M cycles per
forward against a 3.2 M streaming floor). The weight streams run at 1.11 memory beats per clock
at DDR3-1066 (1.04 at DDR3-933: the memory side's own limit after refresh and row switches): each channel's
unpacker and pipe sit in their own 143 MHz clock domain (the trit5 pipe consumes one 128-trit
beat per clock; at the engine's 100 MHz it was the limit) and
each channel's reader in the DDR controller's clock (the 256-bit port at the engine clock was the
cap in between, 0.95). The engine's streaming datapath is kept out of both controllers' I/O clock
regions: next to them its switching narrows the read eye (corrected and the odd uncorrectable
read on controller 0, which the firmware's eye scan does not cover). The token trace (`tprof`) puts the rest at: the engine idle while the scratchpad units
(13%: quantize/rotate, convolution, activations, norms) and the CPU (~8%: list round trips and
records; the descriptor lists are fused across each sub-block - the projection, the residual
add and the next norm run as one list, built while the list before it runs) work on each
layer's serial chain, and the DeltaNet recurrence unit (a 128x128 head in
2,048 clocks at two sub-lanes per channel; at four it sits on its state traffic). Descriptor
lists that do not change between tokens are built once and replayed from an image (the
firmware's list cache); the attention layers' RoPE runs on the norm unit; the attention stream
at 8 lanes / 2 groups runs the 2B's heads in one pass; the attention softmax runs on a
scratchpad unit (exact binary32, the firmware's own exp), so the context costs only the
scores and weighted-V passes: 19.5 tok/s at 256 tokens, 15.8 at 1,024. At DDR3-1066 the ECC
logs the odd corrected read and, rarely, an uncorrectable one that the forward retry repeats
(2 in ~5,500 benched tokens, output exact). openTPU's numbers on the
same card (Qwen3.5-2B 12.1, 4B 5.9 tok/s) are not a like-for-like comparison: the same
architecture, but the original weights at 4 bits (~2.5x the bytes per token) and DDR3-1066.

## Against CPUs and GPUs

Same models (Ternary-Bonsai 1.7B and 8B, batch-1 decode), other hardware:

| platform | memory bandwidth | 1.7B tok/s | 8B tok/s | source |
| --- | ---: | ---: | ---: | --- |
| Apple M4 Pro, CPU (llama.cpp, 10 threads) | 273 GB/s | 123 | 32 | [PrismML 1.7B](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf) / [8B](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf) model cards |
| Apple M4 Pro, GPU (llama.cpp, Metal) | 273 GB/s | 229 | 76 | same (82 in [PrismML's own runtime](https://prismml.com/news/ternary-bonsai)) |
| iPhone 17 Pro Max | | | 27 | [PrismML](https://prismml.com/news/ternary-bonsai) |
| NVIDIA RTX 3080 (CUDA, PrismML's llama.cpp fork) | 760 GB/s | 445 | 199 | measured by us, `llama-bench tg128` (no published NVIDIA figure yet) |
| **TriXium**, FK33 (HBM2) | | **106.3** | **30.3** | this repo |
| **TriXium**, XCU1525 (4x DDR4-1500) | 48 GB/s | 53.8 | 14.9 | this repo |
| **TriXium**, YPCB-00338 (2x DDR3-1066) | 17.1 GB/s | 35.2 | 8.11 | this repo |

A 320 W desktop GPU with 44x the memory bandwidth is, unsurprisingly, much faster. The fairer
reading is per byte of bandwidth, where batch-1 decode lives: on the 8B the YPCB-00338 turns
each GB/s into 0.47 tok/s, against 0.26 for the RTX 3080 and 0.28 for the M4 Pro's GPU - out of
a ten-year-old Kintex-7 and two DDR3 channels. The FK33 runs level with the M4 Pro's CPU (30.3
against 32 tok/s on the 8B, 106 against 123 on the 1.7B), with its HBM far from saturated: there
the engine, not the memory, is the limit.

## Energy per token: estimates for now

Not measured yet. Two estimates bracket it: the FPGA's own power from Vivado's power report for
each build, and the card's power-delivery limit as a worst case. We will refine them with proper
measurements. mWh/token = W / tok/s / 3.6:

| platform | power | 1.7B mWh/token | 8B mWh/token | source |
| --- | ---: | ---: | ---: | --- |
| Apple M4 Pro | measured | | 0.105 | [PrismML](https://prismml.com/news/ternary-bonsai) |
| iPhone 17 Pro Max | measured | | 0.132 | [PrismML](https://prismml.com/news/ternary-bonsai) |
| NVIDIA RTX 3080 | 320 W (board rating) | 0.20 | 0.45 | our tok/s at its rated power (an upper bound) |
| **TriXium**, FK33 | 27.0 W FPGA / 75 W card max | **0.071** / 0.20 | **0.25** / 0.69 | Vivado report / slot limit |
| **TriXium**, XCU1525 | 19.6 W FPGA / 225 W card max | 0.10 / 1.16 | 0.37 / 4.2 | same (27B: 1.4 / 16) |
| **TriXium**, YPCB-00338 | 13.3 W FPGA / 75 W card max | 0.105 / 0.59 | 0.46 / 2.6 | same |

The FPGA figures (2026-10-16), from Vivado's `report_power`: FK33 (`build/fk33`) 27.0 W (25.4
dynamic, 1.7 static), XCU1525 (`build/xcu1525`) 19.6 W (16.7 dynamic, 2.8 static), YPCB-00338
(`build/ypcb_ibuf`) 13.3 W (12.9 dynamic, 0.5 static). Caveats:

- Vivado's estimate uses its default switching activity (low confidence) and covers the FPGA
  only: no DRAM (the HBM2 of the FK33 is in the package and is counted), no regulators, no fans.
  The real card draw is higher than the FPGA estimate; how much higher is what a measurement
  will tell.
- The card limits are what the power inputs allow (PCIe CEM: 75 W from the slot, 150 W more
  from an 8-pin connector, assumed present on the XCU1525 only), not what the cards draw.
- To measure: the XADC/SYSMON supply readings (`trixium_smi info` on builds with the XADC) give
  the FPGA rails; a meter on the slot and auxiliary inputs gives the card.

## Against other FPGA designs

openTPU on the same YPCB-00338 card, and the published FPGA LLM accelerators:
[`related-work.md`](related-work.md).
