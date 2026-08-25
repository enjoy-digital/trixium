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
