# Timeline

From an empty repository to three cards, in dates. What each step taught is in
[`method.md`](method.md); the current numbers are in [`results.md`](results.md).

*Bonsai* is PrismML's model family and the model name (BONS image format, `bonsai-*.img`).
Removed features (BONS v4 native reader, run-ahead prefetch, speculative decoding with MTP at commit `48acf64`) and the
old per-day logs are in git history.

## Steps

| date | step | result |
| --- | --- | --- |
| 2026-07-27 | Repository, FK33 target, BONS format, golden model, ternary/Q8 engine | The Verilator SoC boots the firmware; tokens match the Python reference on day one |
| 2026-07-28/30 | Vector unit (ISA, assembler, interpreter, RTL), offload ladder M0-M8, descriptor sequencer | Every op offloadable, gated by tokens in simulation |
| 2026-07-31/08-03 | The 225 MHz timing campaign | First timing-clean bitstream: 8 channels, 200 MHz |
| 2026-08-03 | First FK33 hardware session | Board configured, HBM up, firmware running; four problems no simulation showed |
| 2026-08-04 | The 1.7B answers " Paris" on the card | 2.033 tok/s |
| 2026-08-06 | Port-matched HBM addressing | 5.555 tok/s |
| 2026-08-28 | VexiiRiscv replaces VexRiscv | 14.54 tok/s |
| 2026-08-30 | Sixteen channels | 35.2 tok/s |
| 2026-09-08 | Result packer (v11), then firmware | 80.6, then 92.0 tok/s |
| 2026-09-10 | 212.5 MHz (v14b) | 101.9 tok/s, the video's climb |
| 2026-09-18 | The 8B on the FK33; the XCU1525 target; the 27B's golden model reproduces PrismML's llama.cpp fork exactly | 8B: 30.0 tok/s |
| 2026-09-19 | FK33 v16b (AXI-legal reset shim); XCU1525 first tokens | 101.2 tok/s; XCU1525 1.7B 5.56 tok/s |
| 2026-09-20 | The 27B token-exact on the XCU1525 | CPU path |
| 2026-09-21 | Architecture review of the 27B | V0 decode baseline 0.217 tok/s |
| 2026-09-22/23 | Resident projection lists (v12), SiLU unit (v14) | 0.363, 0.461 tok/s |
| 2026-09-24/25 | DeltaNet stream (op 5, v17), attention stream (op 6, v18) | 1.84, 3.04 tok/s |
| 2026-09-26/28 | Unit speed-ups and OVERLAP (v19), RoPE on the norm unit (v22) | 3.42, 3.58 tok/s; the area wall |
| 2026-09-29 | The project video ([`illustration/`](illustration/)) | |
| 2026-10-01/06 | DDR4 overclock | DDR4-1600 out of the PHY's reach; DDR4-1500 works: 3.88 tok/s (v30b), 4.03 at sys 137.5 MHz (v31b) |
| 2026-10-05/06 | Speculative decoding (MTP head grafted on the 27B), offline then on the board | Exact, 60-64 % accepted, but 0.72x: removed |
| 2026-10-08 | Release cleanup: TriXium | Checked by equivalence (byte-identical Verilog, sim gate 87/87) |
| 2026-10-10/12 | Board-timing simulation calibrated to 3 %; per-token work cut in firmware; reset-less datapath | FK33 106.3 / 30.3 tok/s (1.7B / 8B) |
| 2026-10-11/13 | The YPCB-00338 port; first card run over JTAG | Both DDR3 controllers level (CK phase), BIST clean; no PCIe link in that host's slot |
| 2026-10-14 | YPCB-00338 on a new host: PCIe up; then eight channels and 5-trit codes | 27.0 / 6.44, then 32.1 / 7.83 tok/s |
| 2026-10-15 | Weight prefetch (engine op 7) and fused vector instructions; demo GIF | 35.2 / 8.11 tok/s. Long replies found not always identical: YPCB DDR3 read margin, XCU1525 marginal DDR4 controller |
| 2026-10-16 | DDR3 ECC (SECDED on the ninth byte lane) on a LiteX-Boards design, then in TriXium; XCU1525 boot-time deep memory test with retraining (development branch) | ECC corrects every injected single-bit fault on the card; five XCU1525 boots in a row clean |
| 2026-10-16 | Per-DQ ECC counters; a DDR3-933 YPCB build | DDR3-1066 errors on controller 1 drift with temperature, a few bits per calibration; DDR3-933 cuts them ~400x (31.8 tok/s on the 1.7B) |
| 2026-10-16 | Docs restructure, a new cut of the video | |
| 2026-10-17 | Ternary-Bonsai-2 2B / 4B on the YPCB-00338 (hybrid build, 4 channels, DDR3-933); ECC upstream; engine writes merged to full DRAM words; forward retry for hybrids (simulation) | 2B 7.41 then 12.40 tok/s, 4B 3.16 then 5.71, tokens exact |

## Measurement ladders

All decode, batch 1, greedy, first tokens checked against the reference.

**FK33, Ternary-Bonsai 1.7B** (N = engine channels):

| date | build | change | cycles/token | tok/s |
| --- | --- | --- | ---: | ---: |
| 2026-08-04 | N=8, 200 MHz | first tokens, HW ops + sequencer lists | | 2.033 |
| 2026-08-06 | | + multi-ARID, port-matched windows | 36.00 M | 5.555 |
| 2026-08-25 | `n8_200_xredx2` | + phased KV, hoisted microcode, xred, loop buffer | 31.19 M | 6.413 |
| 2026-08-28 | `n8_200_vexii3` | VexiiRiscv: 16 KB I-cache, branch prediction | 13.75 M | 14.54 |
| 2026-08-29 | `n8_200_v1` | pipelined vector unit | 7.83 M | 25.5 |
| 2026-08-30 | `n16_200_v1` | 16 channels, multi-store, fused ops | 5.68 M | 35.2 |
| 2026-09-02 | | firmware probe ladder | 3.940 M | 50.8 |
| 2026-09-03 | `n16_200_v6c` | 32 KB I-cache | 3.250 M | 61.5 |
| 2026-09-03 | | ISA rev 2: sixteen vector registers | 3.115 M | 64.2 |
| 2026-09-04 | `n16_200_v8` | 32 KB D-cache | 2.677 M | 74.7 |
| 2026-09-08 | `n16_200_v11` | result packer | 2,480,837 | 80.6 |
| 2026-09-08 | v11 | firmware lists, entry cache, hot-text layout | 2,175,084 | 92.0 |
| 2026-09-10 | `n16_200_v13a` | area campaign | 2,078,083 | 96.2 |
| 2026-09-10 | `n16_212_v14b` | 212.5 MHz | 2,084,447 | 101.9 |
| 2026-09-19 | `n16_212_v16b` | AXI-legal reset shim | 2,100,544 | 101.2 |
| 2026-10-12 | reset-less | Qwen3 residual tail fused, datapath registers reset-less (shipping) | 1,998,309 | 106.3 |

**FK33, Ternary-Bonsai 8B** (212.5 MHz): 7,089k cycles/token = 30.0 tok/s (v15b, 2026-09-18);
7,180,899 = 29.6 tok/s after two correctness fixes; 7,020,640 = 30.3 tok/s (reset-less build,
2026-10-12).

**XCU1525** (decode tok/s; the 27B column is per decoded token unless marked per forward):

| date | build | clocks | 1.7B | 8B | 27B |
| --- | --- | --- | ---: | ---: | --- |
| 2026-09-19 | v3 | sys 125, DDR4-1000 | 5.56 | exact | wrong tokens |
| 2026-09-19 | v4 | sys 150, DDR4-1200 | 23.6 | | |
| 2026-09-20 | v5 | | 43.3 | 10.9 | exact |
| 2026-09-20 | v6 | | 30.2 | | |
| 2026-09-20 | v7 | | 49.586 | | 0.036 bench / 0.055 per forward, CPU path |
| 2026-09-21 | v8 | | | | 680.4 M cycles/forward; V0: 0.217 |
| 2026-09-23 | v12 | | | | 0.363 (411.6 M/forward) |
| 2026-09-23 | v14 | | | | 0.461 (323.7 M/forward) |
| 2026-09-25 | v17 | | | | 1.84 (80.6 M/forward) |
| 2026-09-25 | v18 | | | | 3.04 (49.4 M) |
| 2026-09-26 | v19 | | 50.7 | 13.4 | 3.42 (43.9 M) |
| 2026-09-28 | v22 | | 50.7 | 13.4 | 3.58 (41.9 M) |
| 2026-10-06 | v30b | sys 125, DDR4-1500 | 53.8 | 14.9 | **3.88 (32.2 M)** |
| 2026-10-06 | v31b | sys 137.5, DDR4-1500 | | | 4.03 (34.2 M) |

**XCU1525, the 27B's units** (each bit-exact with the firmware path it replaced, then an
interleaved A/B on one boot):

| step | bitstream | what moved | 27B cycles | decode tok/s |
| --- | --- | --- | ---: | ---: |
| CPU path, baseline | v12, resident off | | 702.4 M/forward | 0.213 |
| resident projection lists: quantize (4), integer Hadamard (3), requantize (5), restore (8) | v12 | FFN down 180.1 M -> 38.7 M (the dense 1,024x1,024 rotations became the FWHT unit) | 411.6 M/forward | 0.363 |
| SiLU unit (11) | v14 | FFN activation 89.97 M -> 0 | 323.7 M/forward | 0.461 |
| DeltaNet stream (engine op 5), K-segment entry cache | v17 | heads 21.6 M -> 3.54 M | 80.6 M/forward | 1.84 |
| attention stream (engine op 6) | v18 | attention 26.8 M -> 5.8 M | 49.4 M/token | 3.04 |
| unit speed-ups, sequencer OVERLAP | v19 | SiLU 17.4k -> 4.9k per layer | 43.9 M/token | 3.42 |
| RoPE on the norm unit (12), per-head recurrence unit dropped | v22 | attention 4.27 M -> 3.34 M | 41.9 M/token | 3.58 |

v30b's 27B profile (cycles per forward): FFN gate/up 9.36 M, FFN down 6.04 M, DeltaNet 10.7 M
(in 1.71, act 2.13, conv 1.87, heads 2.47, out 1.75), attention 2.72 M, logits 0.87 M; 2.34x the
1-beat-per-channel-per-clock floor of 13.5 M. The 27B image is 10,055,180,288 bytes and uploads
in ~112 s.

**YPCB-00338** (sys 125 MHz, DDR3-1066):

| date | build | change | 1.7B tok/s | 8B tok/s |
| --- | --- | --- | ---: | ---: |
| 2026-10-14 | `ypcb_up` | four channels, 2-bit codes | 27.0 | 6.44 |
| 2026-10-14/15 | `ypcb_t5n8c` | eight channels, 5-trit codes | 32.1 | 7.83 |
| 2026-10-15 | `ypcb_pff2` | weight prefetch, fused vector instructions | 35.2 | 8.11 |

Long context on the FK33 1.7B (v14b, 2026-09-15): 89.8 tok/s at 78 tokens of context, 78.7 at
354, 70.0 at 684, 57.4 at 1,344; attention grows by ~0.9k cycles per position per token, and
the vector unit's softmax, not the K/V streams, is the pole.
