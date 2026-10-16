# How it is built: method and lessons

AI was used as a tool, not to generate a one-shot project nobody can maintain. The code was
written by a Claude agent guided by Enjoy-Digital, which set the direction, approved
out-of-spec trade-offs and ran the hardware (OpenAI Codex sessions contributed external
reviews on 2026-09-07 and some of the 2026-09-20/22 work). The method is the one we have
always followed: **experiment, get a first working version, understand it, refine it, strip it
down, and keep solid foundations** for the next step. The dates are in
[`timeline.md`](timeline.md).

![From reference model and simulation to hardware measurements, improvements and upstream contributions](images/experimentation.svg)

## Method

- **Simulation first.** Python models and Migen tests for each module, compiled Verilator unit
  tests, then the full SoC in Verilator with the real firmware on a tiny model with the
  target's structure ([`simulation.md`](simulation.md)). The board answers what simulation
  cannot: timing closure, the board self-tests, real-model tokens and speed.
- **A bit-exact mirror.** The Python mirror reproduces the firmware's arithmetic, including the
  CPU's own float paths: the target RV32F `expf` with its fused operations had to be modelled
  (an independent QEMU oracle linked against the same libm agreed bit for bit on a 20,481-point
  sweep). The simulation gate compares 87 FP32 checkpoints with the mirror; the boards compare
  the same checkpoints read back from the card. A unit landed only when it reproduced the CPU
  path bit for bit.
- **Equivalence, not regression only.** RTL against the interpreter cannot see a microcode
  change (both run the same words), so `test/model/test_asm.py` pins the pre-optimization
  programs; the release cleanup compared emitted Verilog byte for byte.
- **Measurement discipline.** `bench` repeats to 0.0004 % within a boot but the same bitstream
  differed by 5.6 % across deploys (HBM training): a published 6.724 tok/s was retracted when
  the redeploy read 6.365. Since then: A/B within one boot (firmware reload over the running
  bitstream), three reps per bitstream, quote the spread, trust totals over per-op shares.
- **Size a change from its own mechanism, measured on the real model.** The early plan
  projected ~80-85 tok/s from glue measured on the tiny model (2 layers, 256 wide); the card
  measured 13.5 ms of stream and ~166 ms of everything else, because the glue scales with
  layers, width and heads at once. Later, a 2.5-hour bitstream removing a sequencer interlock
  bought 0.005 %, and dual-Q GQA cost 139k LUTs for 0.5 %.
- **A timing gate, not sign-off.** `software/lab/timing.py` refuses a build outside an empirical
  envelope (a reconfigure outside it once took the board host down) and records per-build
  approvals (the OSERDES skew waivers, the `ddr4-overclock` for the XCU1525's ISERDES). A failed
  route is diagnosed by histogramming every failing endpoint by Migen name prefix and logic
  levels, never from the ten worst paths.
- **Unattended board chains.** Build -> timing gate -> configure -> train -> upload -> tokens ->
  bench runs without intervention (`trixium deploy`), under `systemd-run` so a session end does
  not kill a 5-hour build.
- **Tokens before numbers.** A fast wrong answer is not a result.

## Lessons from the climb

- **Measure the real operation, not a proxy.** A single-port `fetch` microbenchmark (~26
  cycles/beat) sent the work towards multi-ARID reads (~5 %); the real fault was addressing
  (channel c's weights read across the HBM switch). Port-matched windows took a matvec from
  ~4.6 to 1.48 cycles/beat: a 3x stream win that bought only 1.30x end to end, because
  streaming was never the majority of the token.
- **The soft CPU was the bottleneck.** A per-cycle PC profiler showed half the CPU cycles
  retiring nothing (63 KB of hot code through a 4 KB I-cache backed by HBM). VexiiRiscv took the
  token from 31.19 M to 13.75 M cycles; the 32 KB I-cache later gave 53 -> 61.5 tok/s.
- **Every write counts.** The vector unit ran 2.9x slower under an engine stream because every
  matvec result was its own scratchpad write; packing eight results per beat (proved 4.96x ->
  1.00x in simulation first) took the card to 80.6 tok/s.
- **When the worst path moves around and is route-dominated, read the utilization report.** The
  225 MHz campaign plateaued at -1.4 to -2.0 ns with the worst path rotating; it was
  congestion. On the XCU1525's VU9P, ~500k LUTs is comfortable and ~585k fails without
  floorplanning.
- **Read the row type of a timing failure.** DDR4-1600 failed at -0.350 ns on ~940 endpoints:
  not skew but the ISERDES Min Period (1.600 ns) of the USPDDRPHY's component-mode bitslices,
  which top out at DDR4-1250 on this -2 part; DDR4-1500 runs as an approved overclock.
- **Speculation needs a cheap token.** The 27B's MTP drafts were 60-64 % accepted on the board,
  but every verified token pays ~10.5 M cycles of per-token work: 0.72x, removed.
- **Exact in simulation is not exact on the card.** Long replies diverged on two boards
  (2026-10-15): the YPCB's DDR3 read window under full streaming is ~2-3 IDELAY taps, and an
  XCU1525 DDR4 controller sometimes trains on a marginal spot and corrupts writes. A 9-token
  check and a 64-beat memory test saw neither ([`boards.md`](boards.md#known-issues)).

## Bugs and lessons

| bug | how it hid | lesson |
| --- | --- | --- |
| Migen's Verilog emission truncates variable shifts, narrow arithmetic in concatenations, negative literals (unsigned compares) | Migen simulation uses Migen semantics; only Verilator and hardware differ. A clamp worked for six milestones, then broke in an unrelated build | No variable shift amounts or negative literals in expressions; signed constants as Signals. Keep end-to-end Verilator checks against a golden model (boot-time VECTEST and per-program CRC sweep) |
| openFPGALoader reports `done 1` on the FK33 while the factory image wins | A bitstream with flipped bytes "loaded" just as well | Program with OpenOCD's `virtex2` pld driver and check the IR capture |
| HBM AXI3: 4-bit AxLEN, 64-beat bursts truncated | The HBM model honoured 8 bits; the test pinned 16 | Check every AxLEN-driving master on AXI3 IP; models truncate like the IP |
| HBM channels aliased through the AXI switch | Upload reported success | Read back distinct per-channel patterns before trusting a layout |
| LitePCIe host-to-device DMA drops data on underflow (the first 2 MB of every upload) | The tool printed its own byte count | Uploads go through BAR0; never trust a count the card did not acknowledge |
| The resident BFP loop lost precision (residual exponent ratchet, fixed `quant8` shift): fluent garbage | The two-layer tiny-model gate passed | Assert precision invariants directly; gate against a multi-layer model |
| Firmware stack in 8 KB SRAM shrank to 1,064 bytes as `.bss` grew | Silent hangs at a place that moved with every relink | A hang that moves with binary layout is memory corruption; the stack now lives in main_ram |
| Vector unit instruction memory wrapped silently past 256 words | One CRC changed in a ten-program sweep | The assembler asserts the budget at build time |
| Interleaved gather consumed and counted results on different conditions | Unit tests stopped at 4 channels and single-row channels | The full design in Verilator on a model sharded like the board |
| Stale incremental verilation mixed RTL versions | Every self-test passed; reverting every source changed nothing | When simulation disagrees with the reference while its self-tests pass, suspect the build |
| Include guards made the host tools compile against the previous build's CSR header | Upload "succeeded" at the old base | Trust only what the tool prints from the card |
| LiteDRAM native-port up-converter: one read in flight | DDR4 sim gated tokens, never rates | Rate tests on the real LiteDRAM core (`test/gateware/memory/test_ddr.py`); the patch is upstream |
| A power-of-two window stride put a DDR4 controller's four channels in the same bank | Correct data, 5.8 cycles/beat | A 32 KB bank-group skew between windows |
| AXI reset shim gated W on AW (illegal for a master) | Worked on LiteDRAM; on the HBM IP each store waited out a timeout (94.9 vs 101.5 tok/s) | The HBM model now accepts all three write-handshake orders |
| Scratchpad (8K x 256) inferred as LUT RAM in the full SoC: v15/v16 at WNS -5.2/-4.8 ns | Synthesized alone it inferred BRAM | Explicit `xpm_memory_sdpram`; grep the synthesis log for "Infeasible attribute" before waiting for place and route |
| AutoCSR exported a public `Memory` as a CSR SRAM with a third port (v13 SiLU tables wrong on the board) | Migen, Verilator unit tests and the SoC simulation all passed | No public `Memory` attributes; `autocsr_exclude` |
| LM-head list stored row chunks contiguously; the image shards rows per channel (v17 wrong logits) | The tiny model's head is one chunk | The simulation gate forces chunking (`lmchunk 128`) |
| K-segment lookups scanned the ~1,100-entry tensor table ~11 times per layer (cached: 110.2 M -> 80.6 M cycles/forward) | Correct output | Suspect any per-layer table lookup; cache it |
| Native reader (v10/v11): 110,555 failing endpoints | Out-of-context synthesis without the board's geometry showed positive slack | Probe new units with the real geometry and read logic levels |
| LiteDRAM native-port CDC drops reads without backpressure | A single read works | Cross at the AXI level; boot every split-clock build in the split simulation |
| DDR trainer passed lanes that wrote nothing | Same seeds, same address: stale data read back as good | Complement patterns; DDRTEST before trusting tokens |
| The MTP head read a KV row nothing wrote | Simulated DDR is zero; the board's was stale | Initialize every buffer a kernel can read |
| No DM pins on the YPCB-00338: 256-bit beats into 512-bit words stored the unstrobed half as zeros | The simulation models honoured the masks | Read-modify-write the partial words; simulation models drop the masks as the board does |
| `wishbone.Cache` reversed the 32-bit words of each L2 line (its default `reverse=True`) | Nothing wrote main_ram over the bus on the other boards | Write over the bus, read with the engine, in a test |
