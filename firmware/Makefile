#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause
#
# TriXium firmware (VexiiRiscv, bare metal), built against a LiteX build tree:
#   make BUILD_DIR=../build/sim_xcu1525   (or ./trixium_sim.py --firmware for the simulation)

BUILD_DIR ?= ../build/sim_xcu1525
PYTHON    ?= python3

include $(BUILD_DIR)/software/include/generated/variables.mak
include $(SOC_DIRECTORY)/software/common.mak

# Flags --------------------------------------------------------------------------------------------

# -O2 instead of common.mak's -Os: at -Os GCC does not inline the small per-head helpers and the
# per-layer hot path still has to fit the 32 KB I$; -O2 -flto measured fastest (-O3 is no better
# and larger).
OPT ?= -O2 -flto -freorder-blocks-and-partition
CFLAGS := $(subst -Os,$(OPT),$(CFLAGS))
# No fused multiply-add: an FPU build (VexiiRiscv --with-rvf) must produce the same float32 bits
# as the soft-float build, and GCC contracts a*b+c into fmadd by default with the F extension.
CFLAGS += -ffp-contract=off

# The portable descriptor encoders shared with the host tools (no MMIO, no board policy).
PORTABLE_LIB = ../software/lib
CFLAGS += -I$(PORTABLE_LIB)
# Bring-up variants, e.g. EXTRA_CFLAGS=-DTRIXIUM_DDR_ONLY (main.c).
CFLAGS += $(EXTRA_CFLAGS)

OBJECTS = crt0.o main.o engine.o model.o infer.o infer_float.o infer_bfp.o infer_lists.o perf.o \
          ops.o vec.o seq.o ddr.o hybrid.o resident.o tokenizer.o storage.o sdload.o sataload.o trixium_sequence.o trixium_tok.o \
          net.o net_ethmac.o web.o

# Build tree stamp ---------------------------------------------------------------------------------

# Objects are tree-specific: the generated CSR map differs between BUILD_DIRs and an object
# built for another tree links fine but reads the wrong registers (the CPU stalls on a bus access
# after "Liftoff!"). The %.o rule only depends on the .c, so this stamp records the tree and the
# CPU flags (soft-float and ilp32f objects do not link together) and wipes the objects on a
# change.
BUILD_STAMP := .build_dir_stamp
-include $(BUILD_STAMP)
ifneq ($(LAST_BUILD_KEY),$(BUILD_DIR) $(CPUFLAGS) $(EXTRA_CFLAGS))
$(shell rm -f *.o *.d trixium.bin trixium.elf)
$(shell echo "LAST_BUILD_KEY := $(BUILD_DIR) $(CPUFLAGS) $(EXTRA_CFLAGS)" > $(BUILD_STAMP))
endif

# Libraries ----------------------------------------------------------------------------------------

COMMA = ,
REGULAR_LIBS = $(filter-out $(ALWAYS_LINK_LIBS),$(LIBS))
ALWAYS_LINK_LIBFLAGS = $(if $(ALWAYS_LINK_LIBS),-Wl$(COMMA)--whole-archive $(ALWAYS_LINK_LIBS:lib%=-l%) -Wl$(COMMA)--no-whole-archive)

# The toolchain's libm for the CPU's ABI (soft-float, or ilp32f with the VexiiRiscv FPU). LiteX's
# -march carries an ISA version (rv32i2p0_m) that GCC 10's multilib matcher does not resolve (it
# silently returns the rv64 default), so the version is stripped here.
LIBM_ARCH := $(subst i2p0_,i,$(patsubst -march=%,%,$(filter -march=%,$(CPUFLAGS))))
LIBM_ABI  := $(patsubst -mabi=%,%,$(filter -mabi=%,$(CPUFLAGS)))

# Targets ------------------------------------------------------------------------------------------

all: trixium.bin

# progs.h (the vector microcode) is regenerated on every build for the encoding of the build
# tree's bitstream (TRIXIUM_VEC_VREGS in soc.h; 16 when absent: a bitstream without the unit). A mismatch is not a
# build error (the unit rejects the words via bad_op, or decodes them as something else), so the
# rule is forced rather than dependency-driven.
VREGS := $(shell grep -oE 'define TRIXIUM_VEC_VREGS [0-9]+' $(BUILD_DIR)/software/include/generated/soc.h 2>/dev/null | grep -oE '[0-9]+$$')
VREGS := $(if $(VREGS),$(VREGS),16)
# Fused vector instructions when the bitstream has them (TRIXIUM_VEC_FUSED in soc.h).
FUSED := $(if $(shell grep -s 'define TRIXIUM_VEC_FUSED' $(BUILD_DIR)/software/include/generated/soc.h),1,0)
progs.h: FORCE
	cd .. && $(PYTHON) -m model.asm firmware/progs.h --vregs $(VREGS) --fused $(FUSED)
FORCE:
infer.o infer_bfp.o infer_lists.o main.o vec.o resident.o: progs.h

# web_page.h: the page web.c serves (web_page.html as one C string).
web_page.h: web_page.html
	$(PYTHON) -c "import sys; d = open(sys.argv[1], 'rb').read(); print('static const char web_page[] = {' + ','.join(str(b) for b in d) + ',0};')" $< > $@
web.o: web_page.h
net.o web.o: CFLAGS += -DTRIXIUM_FIRMWARE   # Their buffers in the main_ram static window.

%.bin: %.elf
	$(OBJCOPY) -O binary $< $@

vpath %.a $(PACKAGES:%=../%)

trixium.elf: $(OBJECTS)
	$(CC) $(LDFLAGS) -T linker.ld -N -o $@ \
		$(OBJECTS) \
		$(PACKAGES:%=-L$(BUILD_DIR)/software/%) \
		-Wl,--start-group \
		$(REGULAR_LIBS:lib%=-l%) \
		$(ALWAYS_LINK_LIBFLAGS) \
		$$($(TARGET_PREFIX)gcc -march=$(LIBM_ARCH) -mabi=$(LIBM_ABI) -print-file-name=libm.a) \
		-Wl,--end-group \
		-Wl,--gc-sections \
		-Wl,-Map,$@.map

# Pull in dependency info for *existing* .o files.
-include $(OBJECTS:.o=.d)

VPATH = $(BIOS_DIRECTORY):$(BIOS_DIRECTORY)/cmds:$(CPU_DIRECTORY):$(PORTABLE_LIB)

%.o: %.c
	$(compile)

%.o: %.S
	$(assemble)

clean:
	$(RM) $(OBJECTS) trixium.elf trixium.bin .*~ *~ *.d *.map

.PHONY: all clean
