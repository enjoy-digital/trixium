# Boards

TriXium runs on two SQRL cards: the **XCU1525** (XCVU9P, 4x DDR4) and the **FK33** (XCVU33P,
HBM2), and the **Inspur YPCB-00338-1P1** (XC7K480T, 2x DDR3). The targets instantiate
the same `TriXiumCore` (16 engine channels on the SQRL cards, 8 on the YPCB-00338);
they differ in the memory system, the clocks and the bring-up.
Builds need Vivado **2024.1** (the 2025.2 install lacks the VU33P).

| | XCU1525 | FK33 | YPCB-00338-1P1 |
| --- | --- | --- | --- |
| FPGA | XCVU9P (3 SLRs) | XCVU33P (1 SLR) | XC7K480T-2 |
| memory | 4x DDR4 x64, 4 GB each (LiteDRAM, USPDDRPHY) | 8 GB HBM2, 16 of 32 pseudo-channels | 2x DDR3 x64 (the ninth, ECC lane used by the ECC build), 2 GB each (LiteDRAM, K7DDRPHY on HR banks, CK phase shift) |
| clocks | sys 125 MHz, DDR4 controllers 187.5 MHz (DDR4-1500) | sys = HBM AXI 212.5 MHz, HBM 800 MHz | sys 125 MHz, DDR3 controllers 133.33 MHz (DDR3-1066) |
| PCIe | Gen3 x4 (LitePCIe) | Gen3 x4 (LitePCIe) | Gen2 x8 (LitePCIe) |
| units | every scratchpad unit, DeltaNet + attention streams | engine, sequencer, vector unit (no scratchpad units) | as the FK33; `--hybrid`: the hybrid models' units, trimmed to fit (4 channels, attention 4 lanes in kv-head slices, Hadamard 8 lanes, norm without RoPE) |
| models | Ternary-Bonsai-2 27B, Ternary-Bonsai 8B / 1.7B | Ternary-Bonsai 8B / 1.7B | Ternary-Bonsai 8B / 1.7B; `--hybrid`: Ternary-Bonsai-2 2B / 4B |
| target | [`trixium_xcu1525.py`](../trixium_xcu1525.py) | [`trixium_fk33.py`](../trixium_fk33.py) | [`trixium_ypcb.py`](../trixium_ypcb.py) |
| build | `./trixium_xcu1525.py --build` | `./trixium_fk33.py --build` | `./trixium_ypcb.py --build` |
| load | `./trixium_xcu1525.py --load` (openFPGALoader) | `./trixium_fk33.py --load` (OpenOCD) | `./trixium_ypcb.py --load` (openFPGALoader, HS2 cable) |
| deploy | `trixium deploy --board xcu1525 --model 27b` | `trixium deploy --board fk33 --model 1p7b` | `trixium deploy --board ypcb --model 1p7b` (hybrid build: `--model 2b`) |

## Known issues

Recording the demo GIF (2026-10-15) showed greedy replies changing from one run to the next. The
design and its simulation are exact; two boards' memory systems are not always:

- **YPCB-00338: DDR3 read errors under full streaming.** Repeating the same matvec, about one
  run in 100 to 300 returned one wrong output row, from DDR3 controller 1's channels in the runs
  analysed. Shifting every bit's read delay while the engine streams shows a clean window of
  only ~2-3 IDELAY taps (~150-230 ps of the 938 ps bit), against the ~10 taps the calibration's
  single-burst checks see; it is worse hot and varies from one calibration to the next. The PHY
  captures reads on the FPGA clock (CK phase mode), so the DRAM's output timing drift under load
  eats the window; on a small, quiet LiteX-Boards design the same card has several taps of
  margin. Full-performance input buffers (`IBUF_LOW_PWR=FALSE`) and the DRAM drive strength
  (34 / 40 ohm) did not change it. **Fix: SECDED ECC on each channel's ninth byte lane** (a
  DFI-level ECC in LiteDRAM, the leveling extended to 72-bit channels). Validated on the card in
  a LiteX-Boards design: every injected single-bit fault corrected, double-bit faults flagged.
  The TriXium build with ECC meets timing (WNS +0.006 ns, 215k LUTs, area unchanged) and runs on
  the card with correct tokens at 35.17 tok/s on the 1.7B. Per-DQ error counters (added to the
  ECC) then showed why ECC alone is not enough at DDR3-1066: under generation load, controller 1
  needs hundreds of thousands of corrections per reply, concentrated on a few bits that change
  with every calibration and drift as the card warms, with enough double-bit errors to defeat
  SECDED; controller 0 stays almost clean. **A DDR3-933 build** (DDR 116.67 MHz, sys 127.27 MHz)
  cuts controller 1's errors about 400x at 31.8 tok/s on the 1.7B; its remaining errors sit on
  two bits, the target of an ECC-guided per-bit retune being evaluated. The ECC is upstream since
  2026-10-17 (LiteDRAM #423, LiteX #2679, LiteX-Boards #928), and the firmware re-runs a forward
  during which a controller flagged an uncorrectable read: the weights in DRAM are intact, the
  errors are transient. A Qwen3 forward is idempotent once its position is restored; a hybrid
  one (Bonsai 2) once its DeltaNet state is double-buffered (engine op 5 writes the other set),
  which is validated in simulation (`eccforce N` re-runs forwards on demand) and next on the
  card.
- **XCU1525: one DDR4 controller trains marginally on some boots.** The weights read back from
  that controller's four channels differ from the image (stably, so the writes are wrong), and
  its KV-cache writes make attention vary from run to run; about half the boots of 2026-10-15
  did it, a different controller each time. The 64-beat `ddrtest` and the BIOS memtest
  (controller 0 only) do not see it. **Fix, on the development branch:** a boot-time deep
  write/read test (`ddrdeep`, 512 random blocks per channel, before the upload) that retrains a
  failing controller until it passes, plus a per-channel weight checksum (`wsum`) against the
  image; five boots in a row came up clean, one of them after catching and retraining
  controller 1.
- **The deploy's token check** compares the first 9-16 tokens; it will gain a long greedy
  generation repeated several times, which is what catches both issues above.

## Build

```sh
./trixium_xcu1525.py --build                    # -> build/xcu1525/, build/xcu1525.verdict
./trixium_xcu1525.py --build --sys-clk-freq 137.5e6 --output-dir build/xcu1525_137
./trixium_fk33.py --build                       # -> build/fk33/, build/fk33.verdict
```

The targets' defaults carry the shipping recipe: the CPU configuration (VexiiRiscv, 32 KB
I$/D$, branch prediction, Zbb, Zcbm, the FPU on the XCU1525), the Vivado directives the designs
closed with, Vivado 2024.1 (`--vivado-settings`, or `LITEX_ENV_VIVADO`) and the driver headers
of the build (`build/<name>/driver/`). The exit code goes to `build/<name>.verdict`: a
successful Vivado run is not enough, the timing gate ([`software/lab/timing.py`](../software/lab/timing.py))
on the routed report decides (the XCU1525 prints the strict verdict for information and ships
on `--experiment` under the DDR4 approval below, `--timing-approval`; the FK33 on the default
gate). A VU9P route needs ~15 GB of disk and ~25 GB of RAM, a VU33P route ~11-15 GB of RAM;
`trixium validate` builds both in parallel.

Target knobs (`./trixium_xcu1525.py --help`, `./trixium_fk33.py --help`):

| target | option | default |
| --- | --- | --- |
| xcu1525 | `--sys-clk-freq` | 125e6 |
| xcu1525 | `--ddr-clk-freq` (4:1, data rate 8x) | 187.5e6 (DDR4-1500) |
| xcu1525 | `--engine-channels` (multiple of 4) | 16 |
| xcu1525 | `--cpu-slr` (SLR of the CPU pblock) | 1 |
| fk33 | `--sys-clk-freq` (= HBM AXI clock) | 212.5e6 |
| fk33 | `--engine-channels` (pseudo-channels 1..N, max 31) | 16 |
| ypcb | `--sys-clk-freq` | 125e6 |
| ypcb | `--ddr-clk-freq` (1:4, data rate 8x) | 133.33e6 (DDR3-1066) |
| ypcb | `--engine-channels` (multiple of 2; the bank skew follows) | 8 |
| ypcb | `--trit5` / `--no-trit5` (5-trit weight codes) | on |
| ypcb | `--prefetch` (weight prefetch: reader FIFO beats per channel, 0 = none) | 2048 |
| ypcb | `--vec-fused` / `--no-vec-fused` (fused vector instructions) | on |
| ypcb | `--read-time` / `--write-time` / `--refresh-postponing` | 256 / 128 / 1 |

### The timing gate

[`software/lab/timing.py`](../software/lab/timing.py) turns the routed timing report into a verdict
(the targets run it after a build, `trixium deploy` again before configuring the card; by hand:
`python3 -m software.lab.timing build/fk33/gateware/sqrl_fk33_timing.rpt`):

- default: SHIP when the misses are no worse than the builds that have run on the card
  (the FK33's gate: its shipping v16b has WNS -0.315 ns, TNS -438.5 ns).
- `--strict` (full closure): setup and hold closed, pulse-width/skew family closed or within a
  named baseline (`--serializer-baseline v8|v8-exact`: the OSERDESE3 CLK/CLKDIV
  skew of the DDR4 PHYs misses by a few picoseconds on every XCU1525 build).
- `--experiment` (board tests of a concept): setup WNS >= -0.75 ns and at most 20,000 failing
  endpoints, hold met. Outside this envelope it refuses and `--force` cannot override it: a
  build at -0.82 ns with ~87k failing endpoints took the host down on reconfigure.
- `--approval <name>`: a per-build exception. `ddr4-overclock` covers the DDR4 overclock
  below.
- `--board <name>` (any mode): the clock periods the board's report must time
  (`REQUIRED_CLOCKS`); a missing one refuses the build before any verdict. The YPCB's target
  and deploy pass `--board ypcb`: its first build had the 50 MHz input unconstrained, and the
  summary (the PCIe clocks alone) passed the strict gate.

**DDR4-1500 is an approved overclock.** LiteDRAM's USPDDRPHY uses component-mode ISERDESE3
whose minimum period on the -2 VU9P is 1.600 ns (625 MHz, i.e. DDR4-1250). At DDR4-1500
(1.333 ns) the report shows a Min Period violation on ~940 serializer clock pins; it is a
datasheet limit, not closable timing. The board trains and runs exact at 1500 (read eyes
132-152 taps); at 1600 the eyes close (44 taps) and one controller does not train. The
build and the deploy pass `ddr4-overclock` (`--timing-approval`, `trixium deploy
--approval`); more bandwidth needs a native-mode PHY.

## XCU1525

![XCU1525: LitePCIe Gen3 x4, 16 engine ports and four LiteDRAM DDR4 controllers](images/board-xcu1525.svg)

### Memory layout

Engine channel `c` streams from controller `c // 4` through its own 512-bit crossbar port (an
AXI bridge that merges two 256-bit beats per controller word,
[`gateware/memory/ddr.py`](../gateware/memory/ddr.py)), crossed from the `ddr` clock domain to
`sys` at the AXI level. Per controller (4 GB): `main_ram` at 0..256 MB on controller 0 (the
CPU, through the L2), the BONS header staging area below 320 MB, then four 704 MB channel
windows at a 1 GB stride, each offset by a 32 KB bank-group skew so the four streams of a
controller never share a DDR4 bank. The last window loses 96 KB to the skew: XCU1525 images
are converted with that stride and reserve.

```sh
trixium convert --board xcu1525 models/8b/Ternary-Bonsai-8B-PQ2_0.gguf        # -> models/bons/bonsai-8b-n16-xcu1525.img
trixium convert --board xcu1525 models/27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf   # -> models/bons/bonsai-27b-n16-xcu1525.img
```

The 1.7B image is the FK33's (`trixium convert`), byte-identical. The 27B GGUF
(`Ternary-Bonsai-2-27B-PTQ1_0.gguf`, PrismML) goes in `models/27b/` with its
`tokenizer.json` (`trixium tokenizer models/27b/tokenizer.json`); `trixium models` fetches
the 1.7B and 8B only.

### DDR4 calibration

LiteX's liblitedram cannot calibrate this board on its own: it reads back what it wrote and
cannot tell a wrong read delay from a wrong write delay, and some byte lanes come out of write
leveling at the edge of their eye. Three pieces fix it:

- LiteX's liblitedram DDR4 MPR read leveling (upstream since October 2026): read leveling on
  the DDR4 MPR patterns (no writes) before write latency calibration. The BIOS and
  `firmware/ddr.c` both run it.
- The target sets `write_dq_dqs_training` on the PHYs (liblitedram's DQ-DQS centring).
- [`software/host/rdlevel.py`](../software/host/rdlevel.py), run on the host over BAR0: a write-side search with the reads
  trusted (DQS against CK, write bitslip, DQ against DQS), centring, then read leveling;
  ~2 s per controller.

On the board host (the deploy copies `software/` there and its boot runs it):

```sh
sudo python3 -m software.host.rdlevel train           # controller 0
sudo python3 -m software.host.rdlevel train --ctrl 2
sudo python3 -m software.host.rdlevel check --ctrl 2  # error bits per module
```

Never train controller 0 while the firmware executes from it (no refresh, no CPU for
~100 ms: the firmware dies). LiteDRAM's 256-bit native up-converter pipelines its reads
(`read_depth`, 8 in flight, upstream); the shipping target uses 512-bit ports, the 256-bit
port path is covered by `test/gateware/memory/test_ddr.py`.

### Deploy

```sh
trixium deploy --board xcu1525 --model 27b                      # bitstream + boot + tokens + bench
trixium deploy --board xcu1525 --model 8b   --no-reconfigure      # next model, same configuration
trixium deploy --board xcu1525 --model 1p7b --no-reconfigure
```

`trixium deploy` ([`software/lab/deploy.py`](../software/lab/deploy.py)) runs from the workstation
against the `[xcu1525]` host of `~/.config/trixium/boards.ini` (format:
[`boards.ini.example`](boards.ini.example); ssh key, sudo through an askpass helper) and
refuses to continue at the first failed step (`--dry-run` prints every command):

1. timing gate (`--experiment` with the approval; the strict verdict is printed alongside);
2. firmware and both header flavours from the build (generated `soc.h` and the driver's
   `kernel/*.h`), host tools and driver built on the host;
3. detach the PCIe endpoint, program with openFPGALoader (`-c ft4232 --fpga-part xcvu9p`),
   verify configuration by the IR capture through openocd (0x35 in each SLR), warm reboot;
4. boot sequence of the board-side session
   ([`software/host/board.py`](../software/host/board.py), `sudo python3 -m software.host.board boot 27b` on the host):
   CPU reset, BIOS calibrates controller 0, host training of controller 0, firmware boot
   without a CPU reset, the firmware calibrates controllers 1-3, host training of 1-3,
   `engreset`, `ddrtest` over the 16 channels;
5. weights (`trixium_load`; check the per-channel base lines), tokens against the reference,
   then `bench` and `stats`.

Reference tokens (greedy):

| model | prompt ids | expected continuation |
| --- | --- | --- |
| 1.7B | `785 6722 315 9625 374` | `12095 11 323 279 6722 315 17689 374 24081` |
| 8B | `785 6722 315 9625 374` | `12095 13 576 6722 315 9625 374 12095 13 576` |
| 27B | `760 6511 314 9338 369` | `11751 13 198 760 6511 314 9564 369` (PrismML's llama.cpp) |

### Rules

- **JTAG is exclusive.** A leftover `openocd` (e.g. from `litex_server --jtag`) holds the
  FT4232H: openFPGALoader then fails, the card keeps the old bitstream and the boot trains a
  removed device. Kill stray openocd processes by PID before a deploy. openocd 0.11 cannot
  program the three-SLR VU9P (it reports "loaded" with every SLR unconfigured).
- **DDRTEST before tokens.** Wrong tokens after DDRTEST errors: re-upload the weights first,
  do not debug the firmware. At sys 137.5 MHz controller 1 shows ~160 errors per channel
  after training (tokens still exact); at 125 MHz 1-2.
- **The PCIe rules of the FK33 below apply here too** (detach, warm reboot, `cpu_rst`).

### Timing (2026-10-12 analysis)

The release-tree build missed the deploy floor in `ddr_clk` (-0.966 ns, 8,949 endpoints; sys
+0.042 ns). Its netlist is v30b's (the Verilog differs only in statement order and the BIOS ROM
size), so the regression is placement, not logic: v30 routed at -1.399 ns, v30b at -0.423 ns
with the same design. Failing families, from the routed checkpoint (all net-dominated: 4.4 ns
of route for 0.8 ns of cells on average, period 5.33 ns):

| family | endpoints | worst | why |
| --- | ---: | ---: | --- |
| ddr reset synchronizer (one flop, 1,401 pins over three SLRs) | 2,673 | -0.89 | fan-out |
| LiteDRAM bankmachine -> multiplexer `choose_cmd` -> `cmd_ready` loop | ~1,700 | -0.95 | 15-17 levels, spread |
| AXI-to-native bridge `w_buffer_level` -> `wdata.ready` -> FIFO read | ~1,500 | -0.93 | 11-17 levels, SLR2->1 |
| DDR CSR bus `s_adr` -> PHY IDELAY/ODELAY CE | ~600 | -0.83 | 2-3 levels, an SLR away |

Controllers 0 and 3 have their I/O in SLR0 and SLR2 (banks 41-42, 70-72), 1 and 2 in SLR1
(65-67, 46-47); the placer puts all four controllers in SLR1 with the CPU, and the TriXium core
adds 184k LUTs there (SLR1 55 % full, SLR0/SLR2 38 %).

Isolation (the SoC with the core replaced by an AXI traffic stub, `ddr_clk` alone): -0.190 ns,
the CSR family only. Hence three fixes, in the RTL:

1. the ddr reset through a replicated register tree (`_CRG`, `max_fanout`);
2. the DDR CSR bus with two registered hops per controller (`PipelinedDDRCSR(hops=2)`);
3. each IDELAY/ODELAY CE registered next to its delay (`register_delay_controls`): the CSR
   strobe still fanned out to ~170 delays over two or three banks after fix 2.

With all three the isolated DDR domain closes (+0.073 ns, strict gate PASS); its limiters are
the two LiteDRAM loops above. In the full design they still miss (-0.925 ns after route): the
core's crowding of SLR1 decides. Tried and refused:

- per-controller SLR pblocks by cell name: the controllers' FSMs and crossbar shift registers
  carry other names, the pblocks split each controller across SLRs (-1.770 ns, DDR-only);
- an exclusive pblock for every `ddr_clk` cell (64.8k cells, 15 clock regions of SLR1): pulls
  controllers 0 and 3's PHY logic away from their I/O, -6.7 ns post-place.

The core alone (out of context, xcvu9p -2L, 125 MHz, placed and routed) closes at +0.039 ns: its
thinnest paths were the scratchpad's BRAM output into the attention stream's 24 operand LUT
RAMs (zero logic levels, ~1,640 paths under +0.08 ns), now registered.

**The bridge loop, broken (2026-10-13).** LiteDRAM's AXI-to-native write path
(`LiteDRAMAXI2NativeW`) now gates on registers: `can_write` comes from one counter of the
buffered write data no command has claimed yet (a flop, where it was a compare of the FIFO level
with the reservation counter), and the write data is released from the registered reservation
level alone (the controller asks for a command's data at least one cycle after accepting it, so
the same-cycle `cmd.ready` term was never needed). Exactly equivalent: LiteDRAM's AXI tests, the
bit-exact gate, the random-init gate with a mid-run soft reset, the FK33 and YPCB simulations
and the fast tier pass. Effect:

| build | ddr_clk WNS | worst ddr path |
| --- | ---: | --- |
| DDR-only, three fixes above | +0.073 ns | the bridge: `w_buffer_level` -> write FIFO BRAM read enable |
| DDR-only, + the bridge | +0.080 ns | PHY register -> OSERDES/ODELAY (0-2 levels, I/O placement); no bridge or bankmachine path in the worst ones |
| full XCU1525, all fixes | **+0.098 ns** | sys **+0.048 ns**, 0 failing endpoints: the strict release gate passes |

This needs LiteDRAM with that change, upstream since LiteDRAM #417 (the registered AXI write
gating). Without it the build is the -0.925 ns one above.

Still unused, should the margin shrink (the fused prepare adds ~4.3k LUTs): the core-exclusion
pblock (the core kept out of a region next to the DDR I/O, the PHYs staying by their banks), a
registered `cmd_ready` in LiteDRAM's crossbar, DDR4-1400 (175 MHz, +0.38 ns of period, ~5 % of
the 27B's rate).

## FK33

![FK33: LitePCIe Gen3 x4, 16 direct HBM2 engine ports and a separate CPU memory port](images/board-fk33.svg)

### Safety

- **Mount a fan over the heatsink before power-on.** The card is passively cooled and
  expects server airflow; its VCCHBM regulator is undersized (the design uses 16 of the 32
  HBM ports at 800 MHz). `trixium_smi info` prints the FPGA temperature: stay below 85 °C,
  abort above 95 °C.
- The bitstream lives in SRAM only: **warm reboots only** after programming; a power cycle
  loses the design.

### Programming: openocd, not openFPGALoader

openFPGALoader reports `done 1` but does not configure the FK33: its JPROGRAM restarts the
board's SPI boot and the factory image wins (`lspci` shows `1e24:1533` instead of `10ee:`,
every CSR reads 0).

```sh
./trixium_fk33.py --load                        # OpenOCD pld load of build/fk33, ~40 s
```

`--load` uses litex-boards' OpenOCD programmer for the FK33 (onboard FT2232,
`prog/openocd_xcvu33p_ft2232.cfg` in litex-boards).

### Deploy

```sh
trixium deploy --board fk33 --model 1p7b
trixium deploy --board fk33 --model 8b
```

Same sequence as the XCU1525 without the DDR4 steps: timing gate, firmware and both header
flavours, driver built on the host, detach + `openocd pld load` + warm reboot, `trixium_load`,
`trixium_chat -k` (CPU reset, firmware through the loader, `TRIXIUM-READY`), tokens, then
`bench 64` + `stats`. Same reference tokens as the XCU1525 table. Reference: 1,998k
cycles/token, 106.3 tok/s on the 1.7B at 212.5 MHz.

## YPCB-00338-1P1

![YPCB-00338: LitePCIe Gen2 x8, eight engine ports and two LiteDRAM DDR3 controllers](images/board-ypcb.svg)

Inspur's YPCB-00338-1P1 (XC7K480T-FFG1156-2, two 72-bit DDR3 channels of nine MT41K256M8, PCIe
Gen2 x8; documented by TiferKing, litex-boards `ypcb_00338_1p1`). openTPU (FeSens) also runs on
this card, with different trade-offs ([`related-work.md`](related-work.md#opentpu-on-the-same-card)).
TriXium's targets: Ternary-Bonsai 1.7B >= 32 tok/s (0.457 GB streamed
per token), 8B >= 7 tok/s (2.010 GB), against the two channels' 17.06 GB/s peak.

Status (2026-10-15): both targets met on the card, tokens checked: the 1.7B decodes at **35.2
tok/s** and the 8B at **8.11 tok/s** with eight engine channels, 5-trit weight codes, the
engine's weight prefetch and the fused vector instructions (the defaults; below). Eight channels
and 5-trit codes alone gave 32.1 / 7.83 tok/s; the first run, four channels and 2-bit
codes, gave 27.0 / 6.44 tok/s. An earlier host never trained a PCIe link with any
image: its slot, not the design.

| | |
| --- | --- |
| area (placed) | prefetch + fused (`build/ypcb_pff2`): 216.0k LUTs (72 %), 856.5 BRAM tiles (90 %); 208.8k LUTs (70 %), 151.0k FFs, 772.5 BRAM tiles (81 %), 558 DSPs (`build/ypcb_t5n8c`); four channels, 2-bit: 126.4k LUTs, 489.5 BRAM tiles |
| timing | prefetch + fused (`build/ypcb_pff2`, 2026-10-15): WNS +0.004 ns, all clocks met (the RMW write-path register of litedram #422, the engine's absmax tree registered; without them -0.829 ns); eight channels, trit5 (`build/ypcb_t5n8c`, 2026-10-14): WNS -0.154 ns on 1,065 endpoints (sys and ddr), runs clean on the card; four channels, sys 125 MHz (`build/ypcb`, 2026-10-11): WNS +0.001 ns, WHS +0.017 ns, pulse width met, 359k endpoints, the strict gate with the clock coverage passes; sys 150 MHz misses by -1.3 ns in the vector unit and the engine (thousands of endpoints) |
| simulation | `--board ypcb`: boot, self-test checksums, tokens against the mirror (both legs), `ddrtest` over the four windows; the calibration itself in the SoC on a fly-by channel model (`--ckphase-model`, `ddrinit 1`) |
| projected decode | 1.7B **25.8 tok/s**, 8B **6.65 tok/s** at sys 125 MHz, DDR3-1066 (below) |
| measured decode | 1.7B **35.2 tok/s**, 8B **8.11 tok/s** (`build/ypcb_pff2`, 2026-10-15, bench 64 / 32); without prefetch/fused 32.1 / 7.83 (`build/ypcb_t5n8c`); four channels, 2-bit: 27.0 / 6.44 (`build/ypcb_up`) |

**The projection.** `trixium_sim.py --board ypcb` with truncated Ternary-Bonsai images (the
models' full widths and LM head, 1 and 2 layers, a 1,024-token KV budget, no int16 embedding
table), on the board's memory system: LiteDRAM's controllers with the board's DDR3-1066 timings
and controller settings, the K7DDRPHY's read/write latencies (CK phase mode), the real 8-bank / 8 KB-row
geometry and bank skew, the board's sys/ddr clock ratio (`--sys-clk-freq`). The 2-layer decode
token minus the 1-layer one is a layer; a model's token is the 1-layer token plus its other
layers (short context, as the FK33/XCU1525 references):

| model | sys | 1 layer | 2 layers | per layer | cycles/token | tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1.7B (28 layers) | 150 MHz | 1,015,501 | 1,185,946 | 170,445 | 5,617,516 | 26.7 (27.0) |
| 1.7B | 133.33 MHz | 915,490 | 1,071,974 | 156,484 | 5,140,558 | 25.9 (26.2) |
| 1.7B | 125 MHz | 865,629 | 1,015,095 | 149,466 | 4,901,211 | 25.5 (25.8) |
| 8B (36 layers) | 150 MHz | 2,243,217 | 2,806,449 | 563,232 | 21,956,337 | 6.83 |
| 8B | 125 MHz | 1,905,156 | 2,388,156 | 483,000 | 18,810,156 | 6.65 |

(In parentheses: the card's 1.7B image keeps its int16 embedding table, a row fetch instead of
the ~60k cycles of CPU decode the truncated images pay per token; the 8B image has none either
way.) The FFN projections stream at 0.80-0.82 beats per channel per sys clock at 150 MHz,
90-92 % of the two controllers' peak (0.889 at 150/133.33 MHz); the LM head at 0.83. About a
quarter of a layer is non-streaming work (norms, attention, quantization, list setup: the
vector unit and the CPU, serial with the streams): the 1.7B target needs it hidden under the
streams (the streams alone: ~4.4M cycles, ~34 tok/s at 150 MHz). Measured levers: eight engine
channels (four windows per controller) stream the projections 3.5 % faster but leave a layer's
cycles unchanged (the 1-layer token: 984k against 1,016k); the controller settings move the token by
less than 0.25 % (`ddr_controller_settings()`); the sys clock only shortens the non-streaming
quarter.

### Memory layout

Engine channel `c` streams from controller `c // 2` through its own 512-bit crossbar port (the
XCU1525's AXI bridge and crossing, [`gateware/memory/ddr.py`](../gateware/memory/ddr.py)). Each
controller is a 64-bit DDR3 channel (eight x8 chips; the ninth, ECC, lane is not used: the
data path is 512 bits per controller word) of 2 GB. Per controller: `main_ram` at 0..256 MB on
controller 0, the BONS header staging area below 320 MB, then two 704 MB channel windows at a
1 GB stride, the second offset by a 32 KB bank skew.

The skew is DDR3 arithmetic: LiteDRAM's ROW_BANK_COL mapping on a 64-bit channel (1024 columns,
8 KB rows) puts the bank in byte address bits 13..15, so a stream changes bank every 8 KB and
goes round the 8 banks every 64 KB. Both windows of a controller stream the same relative
addresses at the same time (one descriptor drives every channel); 32 KB keeps them four banks
apart, so neither crossbar port waits on the other's bank lock nor thrashes its row. The second
window loses the 32 KB (the converter's window reserve).

```sh
trixium convert --board ypcb models/1.7b/Ternary-Bonsai-1.7B-PQ2_0.gguf   # -> models/bons/bonsai-1p7b-n8-ypcb.img
trixium convert --board ypcb models/8b/Ternary-Bonsai-8B-PQ2_0.gguf       # -> models/bons/bonsai-8b-n8-ypcb.img
```

`--board ypcb` defaults to 8 channels and 5-trit codes (`--channels 4 --no-trit5` for a 2-bit,
four-channel bitstream). At 8 channels each controller splits evenly into four 432 MB windows.
Per channel: the 1.7B is 49 MB of weights + 18 MB of KV + its 78 MB int16 embedding table
(126 MB payload, 1.01 GB image); the 8B 229 MB + 24 MB of KV + 155 MB of embedding table
(382 MB payload, 3.06 GB image).

### DDR3 on HR banks: PHY and calibration

The DDR3 banks (11-18) are high-range: no ODELAY, so LiteDRAM's 7-series PHY cannot write-level,
and each channel's nine chips hang on a fly-by command/clock bus. With one static DQS phase
only part of the lanes meet tDQSS. Both channels run at full width with write leveling by CK
phase instead. The HR-bank approach follows the YPCB-00338 bring-up documented by openTPU (FeSens).

Both halves are upstream (developed here, validated on the card, then contributed):

- The PHY: LiteDRAM's `K7DDRPHY(with_odelay=False, with_ck_phase_shift=True)` (LiteDRAM #418):
  the S7DDRPHY's structure with CK and the command pins on MMCM outputs with fine phase shift,
  DQ, DQS (+90 deg) and the read capture on static outputs of the same MMCM; one MMCM per
  channel, cascaded from the controller clock with its feedback through a BUFG; the commands
  cross into the shifted clock through falling-edge registers; a per-lane write latency
  (bitslip) and a per-bit read framing (`dq_dly_sel`).
- The calibration: liblitedram's CK phase leveling (LiteX #2670, #2671; `SDRAM_PHY_CKPHASE_STEPS`
  from the PHY), run by the SoC's own CPU (no host trainer): the CK phase is scanned over one tCK
  (fine steps of 16.7 ps); at each step a full calibration (JEDEC init, per-lane write latency,
  per-bit read leveling) and a LiteDRAM BIST under traffic (a single-burst DFI check is not
  enough); the channel settles in the middle of the longest run that every lane and the BIST
  pass. The BIOS levels both controllers at boot (`SDRAM_CSR_STRIDE`, `SDRAM_CONTROLLERS`);
  the firmware's `ddrinit 1` re-levels controller 1 (`sdram_ckphase_leveling(1)`).
- The simulation keeps a model of the PHY's CSRs and a fly-by channel
  ([`gateware/memory/ckphase_model.py`](../gateware/memory/ckphase_model.py)), checked against
  the PHY's settings (`test_ckphase_model.py`).

### Deploy

```sh
trixium deploy --board ypcb --model 1p7b
trixium deploy --board ypcb --model 8b --no-reconfigure
```

The XCU1525's sequence without the host training: strict timing gate, firmware and both header
flavours, driver built on the host, detach + OpenOCD `pld load` (DONE checked in the IR capture,
0x35; openFPGALoader before v0.12 does not know the XC7K480T) + warm reboot,
then `software/host/board.py boot --board ypcb`: CPU reset, the BIOS levels controller 0
(`CK phase leveling (controller 0)` ... `OK (BIST ...)`), passes its memtest and levels
controller 1, firmware boot without a CPU reset (`trixium_chat -R`), `engreset`, `ddrtest` over
the four channels; then the weights, tokens (the reference table above) and `bench`. JTAG: an FT232H
(Digilent HS2) cable on the card's header shows the FPGA alone; a Xilinx Platform Cable's chain
also has the card's CPLD (`--index-chain`).

### Eight channels and 5-trit codes

**2026-10-15, `build/ypcb_t5n8c`** (`./trixium_ypcb.py --build`: eight engine
channels, four windows per controller, the 5-trit unpackers, litedram #420/#421 for the ddr-domain
paths). `trixium deploy --board ypcb`, the images of `trixium convert --board ypcb`:

| | 1.7B | 8B |
| --- | --- | --- |
| tokens | the reference | the reference |
| decode (cycles/token) | 3,890,287 | 15,973,678 |
| decode | **32.1 tok/s** | **7.83 tok/s** |
| engine busy | 80 % | 85 % |
| four channels, 2-bit (below) | 27.0 | 6.44 |

`ddrtest` passes on all eight windows, `SPTEST` too.

**2026-10-15, `build/ypcb_pff2`**: the same, plus the engine's weight prefetch (op 7, 2048-beat
reader FIFOs) and the fused vector instructions
([`performance-next-steps.md`](performance-next-steps.md)); timing met (WNS +0.004 ns):

| | 1.7B | 8B |
| --- | --- | --- |
| tokens | the reference | the reference |
| decode (cycles/token) | 3,553,018 | 15,407,797 |
| decode | **35.2 tok/s** (+9.6 %) | **8.11 tok/s** (+3.6 %) |
| engine busy | 79 % | 85 % |
| prefetch | 9,546 hits, 0 mismatches (66 forwards) | |

The simulation predicted 34.6 tok/s on the 1.7B slice (the card +1.7 % in tok/s; the
calibration's +1.2 % before). The 8B gains less, as predicted: its gaps outlast the FIFO's
reach. The board-true YPCB simulation
(`trixium_sim.py --board ypcb`, the target's CPU) predicts 31.7 tok/s on the 1.7B slice (+1.2 %
in cycles; [`simulation.md`](simulation.md)).

Against openTPU on the same card: [`related-work.md`](related-work.md#opentpu-on-the-same-card).

### PCIe and first models

**2026-10-14, first PCIe runs.** The
factory image enumerates (10ee:7028, Gen2 x8), and so does TriXium after the JTAG load and a warm
reboot (a rescan alone does not retrain the link): `Version TriXium on YPCB-00338-1P1`, 256 KB
BAR, link 5 GT/s x8. `trixium deploy --board ypcb` then runs the whole sequence in ~5 minutes:

| | 1.7B | 8B |
| --- | --- | --- |
| tokens (first 16) | the reference | the reference |
| decode (cycles/token) | 4,630,085 | 19,421,390 |
| decode | **27.0 tok/s** | **6.44 tok/s** |
| projection (sim, above) | 25.8 | 6.65 |

Both controllers level at every boot as before (CK +48 run 44 / -45 run 78, BIST 0 errors). The
1.7B profile: the engine busy 83 % of the cycles, both units idle 5 %; `silu_mul` is 57 % of the
instrumented vector/CPU time.

### First card run

**Results (2026-10-12/13, the first host, the card's JTAG through an FT232H HS2 cable).** Without
PCIe the console runs over JTAGBone (`litex_server --jtag`; the host's OpenOCD 0.10 wants the old
`interface ftdi` / `ftdi_vid_pid 0x0403 0x6014` syntax), the firmware by serial boot over the
crossover UART (`litex_term crossover --kernel ...`, then a SoC reset):

| | controller 0 (BIOS) | controller 1 (firmware) | openTPU, same card |
| --- | --- | --- | --- |
| CK phase run (fine steps of one tCK) | 44-46 (five boots) | 78 | 45 / 73 |
| write bitslip | 6 on every lane | 0 on every lane | 6 / 0 |
| read window per bit | 9-12 taps | 9-12 taps | 9-11 taps |
| BIST at the chosen phase | 256 MB, 0 errors | 256 MB, 0 errors | |

Then BIOS memtest OK, `ddrtest` OK on the four channels (engine STORE/FETCH), and a line written
over JTAGBone (wishbone, L2) read back in order by the engine.

**On the upstream PHY and leveling (2026-10-13, `build/ypcb_up`, timing met).** The BIOS levels
both controllers at boot: controller 0 CK +48 (run 44), write bitslips 6; controller 1 CK -45 (run
78), write bitslips 0; BIST 256 MB with 0 errors on each, memtest OK. The firmware, serial-booted,
reports both controllers leveled by the BIOS, then `ddrtest` OK on the four channels (0 errors),
SPTEST and VECTEST OK. Two bugs the simulation could not
show, fixed:

- **No DM pins.** The engine's 256-bit beats reach the 512-bit controller word with half the
  strobes; without DM the controller stores the whole word, the unstrobed half as zeros
  (`ddrtest`'s even beats read 0 on every channel). The engine ports read-modify-write their
  partial words (`ENGINE_RMW`, LiteDRAMAXI2Native's `with_read_modify_write`); both YPCB
  simulation models now drop the masks as the board does (`NoDMPHYModel`), and
  `test_engine_port_path_without_dm` shows the loss without RMW.
- **L2 word order.** `wishbone.Cache` reversed the 32-bit words of each line between the bus and
  the DRAM (its default `reverse=True`; LiteX's `add_sdram` uses `False`): JTAG/PCIe writes into
  main_ram landed reversed for the CPU and the engine. Every board with the ddr-domain main_ram
  (the XCU1525 too) had it; nothing wrote main_ram over the bus there.

## Operating a card

- **Detach the endpoint before reprogramming** (`echo 1 > /sys/bus/pci/devices/<bdf>/remove`):
  reconfiguring under a live link is a surprise link-down, and these hosts have no hotplug/AER.
- **Warm-reboot the host after programming.** A rescan enumerates the endpoint but BAR reads
  are all-ones (it never saw PERST#); the FPGA keeps its configuration across a warm reboot.
- **Reset the CPU, not the SoC.** `ctrl_reset` bit 1 (`cpu_rst`, `python3 -m software.host.cpurst`,
  `trixium_chat -k`) reboots the CPU and keeps memory, loader and DMA; bit 0 (`soc_rst`)
  desynchronises the memory AXI and only a reconfigure recovers. The BIOS accepts a serial
  boot only right after a CPU reset.
- **Headers must match the bitstream.** Firmware, driver and host tools compile against the
  build's generated headers (`make -C firmware BUILD_DIR=../build/<build>`,
  `make -C software/user BUILD_DIR=../../build/<build>`, the driver generated in
  `build/<build>/driver/kernel`; `trixium deploy` does all of it); a mismatch uploads to the wrong base or stalls after `Liftoff!` with no error.
- **Secure Boot** blocks the unsigned driver (`trixium driver load` says so): disable it in
  the BIOS.
- **Tokens first.** A fast wrong answer is not a result; after a token failure assume the
  weights are corrupt until `trixium_load` re-uploads them.
- **Stalls.** `SEQSTALL` / `ENGSTALL` / `VECSTALL` print per-channel beat and stall counters;
  the firmware aborts the reply (`GEN ERR`) and soft-resets the engine cluster
  (`ENGRESET ok`), `engreset` does it by hand. `ENGRESET: a port did not drain` needs a
  reconfigure.
- **Measuring.** Boots drift ~4-5 %: compare same-boot, interleaved A/Bs. `trixium_smi -l 1`
  shows engine/vector utilization and GB/s per channel while a model runs.
- **Long jobs** (builds, deploys, `trixium validate`): run them detached with
  `systemd-run --user -E PATH="$PATH" -E HOME="$HOME" --working-directory=$PWD ...`; a
  deploy reboots the board host and drops its ssh session.
