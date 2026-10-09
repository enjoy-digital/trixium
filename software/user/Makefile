# Makefile for the TriXium user tools (derived from LitePCIe's).
#
# BUILD_DIR selects the gateware build whose bitstream is loaded: the generated headers
# (TRIXIUM_N_CHANNELS, CSR map) differ per build and a mismatch silently misconfigures the
# card. Host code compiles against the driver-flavour headers (csr.h/soc.h/mem.h without the
# <generated/...> includes) the build scripts generate next to the build.
BUILD_DIR ?= ../../../build/xcu1525
GEN_INC   := $(BUILD_DIR)/driver/kernel

CFLAGS=-O2 -Wall -g -I$(GEN_INC) -I$(BUILD_DIR)/software/include -I../kernel -Iliblitepcie -MMD -fPIC
LDFLAGS=-g
CC=$(CROSS_COMPILE)gcc
AR=ar

PROGS=trixium_smi trixium_load trixium_chat

all: $(PROGS)

liblitepcie/liblitepcie.a: liblitepcie/litepcie_dma.o liblitepcie/litepcie_flash.o liblitepcie/litepcie_helpers.o
	ar rcs $@ $+
	ranlib $@

trixium_smi: liblitepcie/liblitepcie.a trixium_smi.o
	$(CC) $(LDFLAGS) -o $@ $^ -Lliblitepcie -llitepcie

trixium_load: trixium_loader.o trixium_load.o
	$(CC) $(LDFLAGS) -o $@ $^

trixium_chat: trixium_loader.o trixium_chat.o
	$(CC) $(LDFLAGS) -o $@ $^

clean:
	rm -f $(PROGS) *.o *.a *.d *~ liblitepcie/*.a liblitepcie/*.o liblitepcie/*.d

%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

liblitepcie/%.o: liblitepcie/%.c
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
$(PROGS:%=%.o) trixium_loader.o \
liblitepcie/litepcie_dma.o liblitepcie/litepcie_flash.o liblitepcie/litepcie_helpers.o: $(BUILD_STAMP)
