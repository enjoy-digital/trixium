#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause
#
# Makefile for the TriXium user tools (derived from LitePCIe's).
#
# BUILD_DIR selects the gateware build whose bitstream is loaded: the generated headers
# (TRIXIUM_N_CHANNELS, CSR map) differ per build and a mismatch silently misconfigures the
# card. Host code compiles against the build's generated driver (LitePCIe, name "trixium"):
# its driver-flavour headers (csr.h/soc.h/mem.h without the <generated/...> includes),
# litepcie.h and liblitepcie.
BUILD_DIR ?= ../../build/xcu1525
DRV_DIR   := $(BUILD_DIR)/driver
LIB_DIR   := $(DRV_DIR)/user/liblitepcie

CFLAGS=-O2 -Wall -g -I$(DRV_DIR)/kernel -I$(BUILD_DIR)/software/include -I$(LIB_DIR) -MMD -fPIC
LDFLAGS=-g
CC=$(CROSS_COMPILE)gcc
AR=ar

PROGS=trixium_smi trixium_load trixium_chat

all: $(PROGS)

liblitepcie.a: litepcie_dma.o litepcie_flash.o litepcie_helpers.o
	$(AR) rcs $@ $+
	ranlib $@

trixium_smi: liblitepcie.a trixium_smi.o
	$(CC) $(LDFLAGS) -o $@ $^ -L. -llitepcie

trixium_load: trixium_loader.o trixium_load.o
	$(CC) $(LDFLAGS) -o $@ $^

trixium_chat: liblitepcie.a trixium_loader.o trixium_chat.o
	$(CC) $(LDFLAGS) -o $@ $^ -L. -llitepcie

clean:
	rm -f $(PROGS) *.o *.a *.d *~

%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

litepcie_%.o: $(LIB_DIR)/litepcie_%.c
	$(CC) -c $(CFLAGS) -o $@ $<

-include $(wildcard *.d)

# Switching builds must rebuild the host objects even when the selected headers are older than
# the previous build's objects: a tool with stale CSR addresses must never be reused. The stamp
# content changes only when the configuration does.
BUILD_STAMP := .build_config
BUILD_KEY := $(abspath $(BUILD_DIR))|$(CC)|$(CFLAGS)|$(LDFLAGS)
ifneq ($(strip $(file <$(BUILD_STAMP))),$(strip $(BUILD_KEY)))
$(file >$(BUILD_STAMP),$(BUILD_KEY))
endif
$(PROGS:%=%.o) trixium_loader.o litepcie_dma.o litepcie_flash.o litepcie_helpers.o: $(BUILD_STAMP)
