# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause
#
# VexRiscv bare-metal firmware. Build against a LiteX build tree:
#   make BUILD_DIR=../build/sim   (or ../build/trixium_c)

BUILD_DIR ?= ../build/sim

include $(BUILD_DIR)/software/include/generated/variables.mak
include $(SOC_DIRECTORY)/software/common.mak
# LiteX's common.mak compiles at -Os, under which GCC declined to inline the per-head
# helpers (attn_grp, attn_di_*, kv_scores_br, kv_region_addr) or the cross-file seq_patch:
# attn_patch_slots ran ~14 calls and one iterative div per head, ~300 cycles of compute
# for a dozen operations (2026-09-08 diagnostics). The firmware is ~80 KB; the per-layer
# hot path has to stay inside the 32 KB I$, so this is measured, not assumed. Interleaved
# A/Bs on v11 (tools/fk33_ab2.sh, 4 reps each, tokens OK): -O2 -7.9% vs -Os; -flto a
# further -1.1% (cross-file inlining of the per-layer helpers); -O3 +0.17% = noise, so the
# smaller -O2 stays. Text: -Os 83.6 KB, -O2 90.9, -O2 -flto 100.8.
OPT ?= -O2 -flto -freorder-blocks-and-partition
CFLAGS := $(subst -Os,$(OPT),$(CFLAGS))

OBJECTS = crt0.o main.o engine.o model.o infer.o perf.o ops.o vec.o seq.o


# Objects are tree-specific: the generated CSR map differs between BUILD_DIRs, so an object
# compiled against build/sim and linked into a build/n8_200_* binary (or vice versa) produces
# firmware that reads the wrong register addresses. The failure is not a link error - it boots,
# jumps to 0x40000000 and then the CPU STALLS on a bus access that never completes, with the
# console silent after "Liftoff!". Cost half a sim run to diagnose.
#
# The %.o rule below depends only on the .c, so nothing forces a rebuild when BUILD_DIR moves.
# This stamp does: it records the tree the objects were built for and wipes them on a change.
BUILD_STAMP := .build_dir_stamp
-include $(BUILD_STAMP)
ifneq ($(LAST_BUILD_DIR),$(BUILD_DIR))
$(shell rm -f *.o *.d bonsai.bin bonsai.elf)
$(shell echo "LAST_BUILD_DIR := $(BUILD_DIR)" > $(BUILD_STAMP))
endif

COMMA = ,
REGULAR_LIBS = $(filter-out $(ALWAYS_LINK_LIBS),$(LIBS))
ALWAYS_LINK_LIBFLAGS = $(if $(ALWAYS_LINK_LIBS),-Wl$(COMMA)--whole-archive $(ALWAYS_LINK_LIBS:lib%=-l%) -Wl$(COMMA)--no-whole-archive)

all: bonsai.bin

# progs.h (the vector microcode) is regenerated on EVERY build for the encoding the build
# tree's bitstream uses (BONSAI_VEC_VREGS in soc.h; absent = the original 8-register
# encoding). A mismatch is not a build error - the unit rejects the words via the sticky
# bad_op, or worse decodes them as something else - so this is forced, not dependency-driven.
VREGS := $(shell grep -oE 'define BONSAI_VEC_VREGS [0-9]+' $(BUILD_DIR)/software/include/generated/soc.h 2>/dev/null | grep -oE '[0-9]+$$')
VREGS := $(if $(VREGS),$(VREGS),8)
progs.h: FORCE
	cd .. && python3 -m trixium.model.asm firmware/progs.h --vregs $(VREGS)
FORCE:
infer.o main.o vec.o: progs.h

%.bin: %.elf
	$(OBJCOPY) -O binary $< $@

vpath %.a $(PACKAGES:%=../%)

bonsai.elf: $(OBJECTS)
	$(CC) $(LDFLAGS) -T linker.ld -N -o $@ \
		$(OBJECTS) \
		$(PACKAGES:%=-L$(BUILD_DIR)/software/%) \
		-Wl,--start-group \
		$(REGULAR_LIBS:lib%=-l%) \
		$(ALWAYS_LINK_LIBFLAGS) \
		$$($(TARGET_PREFIX)gcc -march=rv32im -mabi=ilp32 -print-file-name=libm.a) \
		-Wl,--end-group \
		-Wl,--gc-sections \
		-Wl,-Map,$@.map

-include $(OBJECTS:.o=.d)

VPATH = $(BIOS_DIRECTORY):$(BIOS_DIRECTORY)/cmds:$(CPU_DIRECTORY)

%.o: %.c
	$(compile)

%.o: %.S
	$(assemble)

clean:
	$(RM) $(OBJECTS) bonsai.elf bonsai.bin .*~ *~ *.d *.map

.PHONY: all clean
