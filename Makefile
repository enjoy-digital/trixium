#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

PYTHON  ?= .venv/bin/python
PIP     ?= .venv/bin/pip
PYTEST  ?= $(PYTHON) -m pytest

export PYTHONHASHSEED := 0

# Environment ------------------------------------------------------------------------------------

.PHONY: venv
venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"   # numpy tokenizers pyserial + pytest huggingface_hub; the `trixium` CLI

# Models -----------------------------------------------------------------------------------------

.PHONY: models tiny convert
models:
	./scripts/download_models.sh

# The tiny random models the tests and the simulation use (seconds).
tiny:
	for n in 2 4 8 16; do $(PYTHON) -m trixium.model.tinygen --out models/tiny --channels $$n; done
	for n in 2 8; do $(PYTHON) -m trixium.model.tinygen --out models/tiny --channels $$n --hybrid; done
	$(PYTHON) -m trixium.model.tinygen --out models/tiny --channels 8 --hybrid-1k

# GGUF -> BONS image + the chat tool's tokenizer. `make convert GGUF=... CHANNELS=16`.
GGUF     ?= models/1.7b/Ternary-Bonsai-1.7B-PQ2_0.gguf
CHANNELS ?= 16
convert:
	$(PYTHON) -m trixium.cli convert $(GGUF) --channels $(CHANNELS)
	$(PYTHON) -m trixium.cli tokenizer $(dir $(GGUF))tokenizer.json

# Tests ------------------------------------------------------------------------------------------

.PHONY: test test-guarded
test:
	$(PYTEST)

# One process per test file with a memory guard (some files peak at ~10 GB).
test-guarded:
	$(PYTHON) tools/run_tests_guarded.py --out build/tests

# Simulation (Verilator) -------------------------------------------------------------------------
#
#   make sim                 build: headers, firmware, Verilog (~1 min), Verilator (~2.5 min)
#   make sim-run             run it (console on stdin/stdout)
#   make sim-firmware        rebuild the firmware and re-bake it into the built sim (seconds)
#   make sim-gate            traced generation compared with the Python mirror (~17 min)

SIM_BOARD ?= xcu1525
SIM_IMAGE ?= models/tiny/bonsai-tiny-hybrid-1k-n8.img
SIM_DIR    = build/sim_$(SIM_BOARD)

.PHONY: sim sim-run sim-firmware sim-gate
sim:
	$(PYTHON) -m trixium.targets.sim --board $(SIM_BOARD) --image $(SIM_IMAGE) --build-software
	$(MAKE) -C firmware BUILD_DIR=../$(SIM_DIR)
	$(PYTHON) -m trixium.targets.sim --board $(SIM_BOARD) --image $(SIM_IMAGE) --firmware firmware/trixium.bin
	cd $(SIM_DIR)/gateware && bash build_sim.sh

sim-run:
	cd $(SIM_DIR)/gateware && ./obj_dir/Vsim

sim-firmware:
	$(MAKE) -C firmware BUILD_DIR=../$(SIM_DIR)
	$(PYTHON) tools/sim_rebake.py $(SIM_DIR)/gateware $(SIM_IMAGE) firmware/trixium.bin

sim-gate:
	mkdir -p build/sim-gate
	$(PYTHON) tools/card_vs_mirror.py mirror $(SIM_IMAGE) --ids '1 2 3' -o build/sim-gate/mirror.txt
	$(PYTHON) tools/sim_resident_gate.py --sim $(SIM_DIR)/gateware --out build/sim-gate \
	    --ids '1 2 3' --dbg 3 --expect-silu --skip-off
	$(PYTHON) tools/card_vs_mirror.py compare build/sim-gate/mirror.txt build/sim-gate/trace-on.txt

# Gateware (Vivado 2024.1) -----------------------------------------------------------------------

.PHONY: gateware-xcu1525 gateware-fk33
gateware-xcu1525:
	tools/build_xcu1525.sh xcu1525

gateware-fk33:
	tools/build_fk33.sh fk33

# Software ---------------------------------------------------------------------------------------
#
# BUILD: the build directory (under build/) whose headers the firmware/host tools compile against.

BUILD ?= xcu1525

.PHONY: firmware software driver
firmware:
	$(MAKE) -C firmware BUILD_DIR=../build/$(BUILD)

software:
	$(MAKE) -C trixium/software/user BUILD_DIR=../../../build/$(BUILD)

driver:
	$(MAKE) -C trixium/software/kernel

# Cleanup ----------------------------------------------------------------------------------------

.PHONY: clean-vivado clean
clean-vivado:
	rm -rf build/*/gateware/vivado* build/*/gateware/.Xil build/*/gateware/*.dcp
	rm -rf .Xil vivado*.jou vivado*.log

clean:
	rm -rf build
	find . -name __pycache__ -type d -exec rm -rf {} +
