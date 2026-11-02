# AGENTS.md: TriXium

Ternary LLMs on FPGA: PrismML's Bonsai models (Qwen3 1.7B/8B, Bonsai-2 hybrid 2B/4B/27B) decoded
entirely on FPGA cards with LiteX (VexiiRiscv firmware, adder-tree matvec engine, vector and
fixed-function units, LiteDRAM/HBM memory systems); the host only tokenizes.

## Where things are

- Targets: `trixium_fk33.py`, `trixium_xcu1525.py`, `trixium_ypcb.py`, `trixium_ti375.py`, `trixium_sim.py` (the same SoC in Verilator).
- `gateware/` (`core/`, `memory/`, `units/`), `firmware/` (C: engine, inference, chat), `model/` (conversion, bit-exact reference/mirror, GGUF reader).
- `software/` (`trixium` CLI: deploy, chat, smi, dashboard; host library, driver, lab helpers), `test/`.
- `doc/`: `getting-started.md`, `architecture.md`, `boards.md`, `method.md`, `simulation.md`, `results.md`, `units.md`, `timeline.md`.

## Verify a change

```sh
python3 -m pytest -n auto                                           # fast tier
python3 test/run_guarded.py --runslow                               # full tier, in parallel under a memory guard
python3 -m pytest test/system/test_sim_smoke.py --runslow           # one slow test (built simulation)
```

- Method (`doc/method.md`): simulation first (unit, Verilator, full SoC with the real firmware),
  the bit-exact model as the contract, tokens before numbers, A/B comparisons within one boot.
- Hardware runs go through `trixium deploy --board <board> --model <model>` with a local
  `boards.ini` (see `doc/boards.ini.example`); never commit it, it holds host access details.
- Report only numbers measured on the board, with the build and the command.

## Known pitfalls

- Migen/Verilog widths: give intermediates explicit widths; Migen simulation may not show what the
  generated Verilog computes (variable shifts, narrow arithmetic in `Cat`).
- HBM AXI3: 16-beat maximum bursts (longer ones silently corrupt writes).
- LitePCIe host-to-device DMA can drop the first ring on underflow: weight uploads go through BAR0.
- Efinix Ti375: one AXI ID per reader (the hard controller reorders IDs).
- Firmware: a hang that moves with the binary layout is memory corruption (stack in small SRAM).
- Vivado: check that large memories are not inferred as LUT RAM ("Infeasible" in the synthesis log);
  diagnose failed routes by failing endpoints grouped by name, not by the top paths.

## LiteX

This project is built with LiteX: read https://github.com/enjoy-digital/litex/wiki/LiteX-for-AI-Agents (framework map, conventions, how to
verify a change, contribute back the generic fixes to LiteX and its cores, ex: LiteDRAM).

## Debug on hardware

Method: https://github.com/enjoy-digital/litex/wiki/Debug-A-SoC-On-Hardware (check the link and the SoC identity first, registers before
traces, narrow LiteScope captures, read the generated Verilog, turn findings into tests).
`litex_mcp` gives an agent the same loop as MCP tools through `litex_server` (CSRs by name, LiteScope
captures as data, console, builds with parsed timing, simulation; writes and `--load`/`--flash`
only with `--allow-write`): https://github.com/enjoy-digital/litex/wiki/LiteX-for-AI-Agents#drive-a-soc-through-mcp.

## Rules

- LiteX style: https://github.com/enjoy-digital/litex/blob/master/doc/coding_style.md.
- Never edit generated files (`build/`, `csr.h`, `csr.csv`, generated Verilog): change the Python.
- Never write a card's flash unless the user asked (the factory images stay); `--load` for experiments.
- Never commit weights, bitstreams, credentials or host-specific paths.
