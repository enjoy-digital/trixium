# Point BUILD_DIR at the gateware build whose bitstream is loaded: the generated headers
# (TRIXIUM_N_CHANNELS, CSR map) differ per build and a mismatch silently misconfigures the
# card. Defaults to build/xcu1525 (tools/build_xcu1525.sh xcu1525).
BUILD_DIR ?= ../../../build/xcu1525
# The driver-flavour headers (csr.h/soc.h/mem.h without the <generated/...> includes) are
# what host code compiles against; the deploy script regenerates them next to the build.
GEN_INC   := $(BUILD_DIR)/driver/kernel

CFLAGS=-O2 -Wall -g -I$(GEN_INC) -I$(BUILD_DIR)/software/include -I../kernel -Iliblitepcie -I../lib -MMD -fPIC
LDFLAGS=-g
CC=$(CROSS_COMPILE)gcc
AR=ar

PROGS=trixium_util trixium_load trixium_chat

all: $(PROGS)

liblitepcie/liblitepcie.a: liblitepcie/litepcie_dma.o liblitepcie/litepcie_flash.o liblitepcie/litepcie_helpers.o
	ar rcs $@ $+
	ranlib $@

trixium_util: liblitepcie/liblitepcie.a trixium_util.o
	$(CC) $(LDFLAGS) -o $@ $^ -Lliblitepcie -llitepcie

trixium_load: liblitepcie/liblitepcie.a trixium_loader.o trixium_load.o
	$(CC) $(LDFLAGS) -o $@ $^ -Lliblitepcie -llitepcie

trixium_chat: liblitepcie/liblitepcie.a trixium_loader.o trixium_chat.o
	$(CC) $(LDFLAGS) -o $@ $^ -Lliblitepcie -llitepcie

clean:
	rm -f $(PROGS) *.o *.a *.d *~ liblitepcie/*.a liblitepcie/*.o liblitepcie/*.d

%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

liblitepcie/%.o: liblitepcie/%.c
	$(CC) -c $(CFLAGS) -o $@ $<

-include $(wildcard *.d)

# Switching maps must rebuild host objects even when the selected headers are
# OLDER than the previous build's objects. Do not reuse a utility with stale CSR
# addresses. A content-sensitive stamp changes only when the configuration does;
# no deletion of unrelated objects/artifacts is needed.
BUILD_STAMP := .build_config
BUILD_KEY := $(abspath $(BUILD_DIR))|$(CC)|$(CFLAGS)|$(LDFLAGS)
ifneq ($(strip $(file <$(BUILD_STAMP))),$(strip $(BUILD_KEY)))
$(file >$(BUILD_STAMP),$(BUILD_KEY))
endif
$(PROGS:%=%.o) trixium_loader.o caps_parser.o region_upload.o region_read.o \
liblitepcie/litepcie_dma.o liblitepcie/litepcie_flash.o liblitepcie/litepcie_helpers.o: $(BUILD_STAMP)
