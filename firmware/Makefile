# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause
#
# TriXium firmware (VexiiRiscv, bare metal). Build against a LiteX build tree:
#   make BUILD_DIR=../build/sim_xcu1525   (or ../build/<board build>)

BUILD_DIR ?= ../build/sim_xcu1525

include $(BUILD_DIR)/software/include/generated/variables.mak
include $(SOC_DIRECTORY)/software/common.mak
# LiteX's common.mak compiles at -Os, under which GCC declines to inline the per-head
# helpers (attn_grp, attn_di_*, kv_scores_br, kv_region_addr): ~300 cycles of calls and
# divisions per head for a dozen operations. The per-layer hot path has to stay inside the
# 32 KB I$, so this is measured, not assumed: -O2 -7.9% vs -Os; -flto a further -1.1%
# (cross-file inlining of the per-layer helpers); -O3 +0.17% = noise, so the smaller -O2
# stays. Text: -Os 83.6 KB, -O2 90.9, -O2 -flto 100.8.
OPT ?= -O2 -flto -freorder-blocks-and-partition
CFLAGS := $(subst -Os,$(OPT),$(CFLAGS))
# No fused multiply-add: a build with the hardware FPU (VexiiRiscv --with-rvf, the XCU1525's
# 27B CPU path) must produce the same float32 bits as the soft-float sim that gated it, and
# GCC contracts a*b+c into fmadd by default once the F extension is in -march.
CFLAGS += -ffp-contract=off

OBJECTS = crt0.o main.o engine.o model.o infer.o perf.o ops.o vec.o seq.o ddr.o hybrid.o \
          resident.o trixium_sequence.o
# The portable descriptor encoders shared with the host tools (no MMIO, no board policy).
PORTABLE_LIB = ../trixium/software/lib
CFLAGS += -I$(PORTABLE_LIB)


# Objects are tree-specific: the generated CSR map differs between BUILD_DIRs, so an object
# compiled against build/sim and linked into a build/n8_200_* binary (or vice versa) produces
# firmware that reads the wrong register addresses. The failure is not a link error - it boots,
# jumps to 0x40000000 and then the CPU STALLS on a bus access that never completes, with the
# console silent after "Liftoff!".
#
# The %.o rule below depends only on the .c, so nothing forces a rebuild when BUILD_DIR moves.
# This stamp does: it records the tree the objects were built for and wipes them on a change.
# The CPU flags are part of the stamp too: the same tree regenerated for another CPU (the
# FPU sim, ilp32f) leaves objects the linker refuses to mix ("can't link single-float
# modules with soft-float modules") unless everything is rebuilt.
BUILD_STAMP := .build_dir_stamp
-include $(BUILD_STAMP)
ifneq ($(LAST_BUILD_KEY),$(BUILD_DIR) $(CPUFLAGS))
$(shell rm -f *.o *.d trixium.bin trixium.elf)
$(shell echo "LAST_BUILD_KEY := $(BUILD_DIR) $(CPUFLAGS)" > $(BUILD_STAMP))
endif

COMMA = ,
REGULAR_LIBS = $(filter-out $(ALWAYS_LINK_LIBS),$(LIBS))
ALWAYS_LINK_LIBFLAGS = $(if $(ALWAYS_LINK_LIBS),-Wl$(COMMA)--whole-archive $(ALWAYS_LINK_LIBS:lib%=-l%) -Wl$(COMMA)--no-whole-archive)

all: trixium.bin

# progs.h (the vector microcode) is regenerated on EVERY build for the encoding the build
# tree's bitstream uses (TRIXIUM_VEC_VREGS in soc.h; absent = the 8-register encoding). A mismatch is not a build error - the unit rejects the words via the sticky
# bad_op, or worse decodes them as something else - so this is forced, not dependency-driven.
VREGS := $(shell grep -oE 'define TRIXIUM_VEC_VREGS [0-9]+' $(BUILD_DIR)/software/include/generated/soc.h 2>/dev/null | grep -oE '[0-9]+$$')
VREGS := $(if $(VREGS),$(VREGS),8)
progs.h: FORCE
	cd .. && python3 -m trixium.model.asm firmware/progs.h --vregs $(VREGS)
FORCE:
infer.o main.o vec.o resident.o: progs.h

%.bin: %.elf
	$(OBJCOPY) -O binary $< $@

vpath %.a $(PACKAGES:%=../%)

# The toolchain's libm for the CPU's ABI (soft float today; ilp32f with a VexiiRiscv FPU).
# LiteX's -march carries an ISA version (rv32i2p0_m) that GCC 10's multilib matcher does not
# resolve - it silently returns the rv64 default - so the version is stripped here.
LIBM_ARCH := $(subst i2p0_,i,$(patsubst -march=%,%,$(filter -march=%,$(CPUFLAGS))))
LIBM_ABI  := $(patsubst -mabi=%,%,$(filter -mabi=%,$(CPUFLAGS)))

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

-include $(OBJECTS:.o=.d)

VPATH = $(BIOS_DIRECTORY):$(BIOS_DIRECTORY)/cmds:$(CPU_DIRECTORY):$(PORTABLE_LIB)

%.o: %.c
	$(compile)

%.o: %.S
	$(assemble)

clean:
	$(RM) $(OBJECTS) trixium.elf trixium.bin .*~ *~ *.d *.map

.PHONY: all clean
