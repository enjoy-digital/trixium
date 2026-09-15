# Point BUILD_DIR at the gateware build whose bitstream is loaded: the generated headers
# (BONSAI_N_CHANNELS, CSR map) differ per build and a mismatch silently misconfigures the
# card. Defaults to the shipping build (tools/build_fk33.sh).
BUILD_DIR ?= ../../../build/n16_212_v15b
# The driver-flavour headers (csr.h/soc.h/mem.h without the <generated/...> includes) are
# what host code compiles against; the deploy script regenerates them next to the build.
GEN_INC   := $(BUILD_DIR)/driver/kernel

CFLAGS=-O2 -Wall -g -I$(GEN_INC) -I$(BUILD_DIR)/software/include -I../kernel -Iliblitepcie -MMD -fPIC
LDFLAGS=-g
CC=$(CROSS_COMPILE)gcc
AR=ar

PROGS=bonsai_util bonsai_load bonsai_chat

all: $(PROGS)

liblitepcie/liblitepcie.a: liblitepcie/litepcie_dma.o liblitepcie/litepcie_flash.o liblitepcie/litepcie_helpers.o
	ar rcs $@ $+
	ranlib $@

bonsai_util: liblitepcie/liblitepcie.a bonsai_util.o
	$(CC) $(LDFLAGS) -o $@ $^ -Lliblitepcie -llitepcie

bonsai_load: liblitepcie/liblitepcie.a bonsai_loader.o bonsai_load.o
	$(CC) $(LDFLAGS) -o $@ $^ -Lliblitepcie -llitepcie

bonsai_chat: liblitepcie/liblitepcie.a bonsai_loader.o bonsai_chat.o
	$(CC) $(LDFLAGS) -o $@ $^ -Lliblitepcie -llitepcie

clean:
	rm -f $(PROGS) *.o *.a *.d *~ liblitepcie/*.a liblitepcie/*.o liblitepcie/*.d

%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

liblitepcie/%.o: liblitepcie/%.c
	$(CC) -c $(CFLAGS) -o $@ $<

-include $(wildcard *.d)
