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

OBJECTS = crt0.o main.o engine.o model.o infer.o perf.o ops.o vec.o

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
