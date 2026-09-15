#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

PYTHON  ?= .venv/bin/python
PIP     ?= .venv/bin/pip
PYTEST  ?= .venv/bin/pytest

# Vivado 2024.1 is required: the 2025.2 install lacks VU33P device support.
VIVADO_SETTINGS ?= /opt/Xilinx/2024.1/Vivado/2024.1/settings64.sh

# Environment ------------------------------------------------------------------------------------

.PHONY: venv
venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"   # numpy tokenizers pyserial + pytest huggingface_hub; the `bonsai` CLI

.PHONY: df-check
df-check:
	@avail=$$(df --output=avail -BG / | tail -1 | tr -dc '0-9'); \
	echo "Disk available: $${avail}G"; \
	if [ $$avail -lt 8 ]; then echo "ERROR: less than 8 GB free, clean up first (make clean-vivado)"; exit 1; fi

# Models -----------------------------------------------------------------------------------------

.PHONY: models
models: df-check
	./scripts/download_models.sh

.PHONY: tiny
tiny:
	$(PYTHON) -m trixium.model.tinygen --out models/tiny

# The model path: GGUF -> BONS image (~13 s, ~2 GB RAM for the 1.7B) + the C tool's tokenizer.
GGUF     ?= models/1.7b/Ternary-Bonsai-1.7B-Q2_0.gguf
CHANNELS ?= 16
.PHONY: convert
convert:
	$(PYTHON) -m trixium.cli convert $(GGUF) --channels $(CHANNELS)
	$(PYTHON) -m trixium.cli tokenizer $(dir $(GGUF))tokenizer.json

# The card: deploy a build to the lab board (tools/board.env), or chat where the card is.
BUILD ?= n16_212_v15b
.PHONY: deploy chat
deploy:
	$(PYTHON) -m trixium.cli deploy $(BUILD)
chat: software
	$(PYTHON) -m trixium.cli chat

# Tests / Simulation -----------------------------------------------------------------------------

.PHONY: test
test:
	$(PYTEST)

.PHONY: test-slow
test-slow:
	$(PYTEST) --runslow

.PHONY: sim
sim:
	$(PYTHON) sim/bonsai_sim.py

# Gateware ---------------------------------------------------------------------------------------

.PHONY: gateware-a gateware-b gateware-c
gateware-a: df-check
	bash -c "source $(VIVADO_SETTINGS) && python3 trixium.py --variant=a --build"

gateware-b: df-check
	bash -c "source $(VIVADO_SETTINGS) && python3 trixium.py --variant=b --build"

gateware-c: df-check
	bash -c "source $(VIVADO_SETTINGS) && python3 trixium.py --variant=c --build"

.PHONY: load-a load-b load-c
load-a:
	bash -c "source $(VIVADO_SETTINGS) && python3 trixium.py --variant=a --load"

load-b:
	bash -c "source $(VIVADO_SETTINGS) && python3 trixium.py --variant=b --load"

load-c:
	bash -c "source $(VIVADO_SETTINGS) && python3 trixium.py --variant=c --load"

# Software ---------------------------------------------------------------------------------------

.PHONY: firmware
firmware:
	$(MAKE) -C firmware BUILD_DIR=../build/$(BUILD)

.PHONY: software
software:
	$(MAKE) -C trixium/software/user BUILD_DIR=../../../build/$(BUILD)

.PHONY: driver
driver:
	$(MAKE) -C trixium/software/kernel

# Cleanup ----------------------------------------------------------------------------------------

.PHONY: clean-vivado
clean-vivado:
	rm -rf build/*/gateware/vivado* build/*/gateware/.Xil build/*/gateware/*.dcp
	rm -rf .Xil vivado*.jou vivado*.log
	df -h / | tail -1

.PHONY: clean
clean:
	rm -rf build firmware/build
	find . -name __pycache__ -type d -exec rm -rf {} +
