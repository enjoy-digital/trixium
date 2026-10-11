#
# This file is part of TriXium.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause
#
# libtrixiumddr: the BIOS side of the YPCB's DDR3 calibration (sdram_custom_init() for
# controller 0, firmware/ddr_s7hr.c), a LiteX software package the builder compiles before the
# BIOS and links into it (trixium_ypcb.py).

TRIXIUM_BIOS_DIRECTORY := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))

include ../include/generated/variables.mak
include $(SOC_DIRECTORY)/software/common.mak

CFLAGS += -I$(TRIXIUM_BIOS_DIRECTORY)/..

OBJECTS = bios_ddr.o

all: libtrixiumddr.a

libtrixiumddr.a: $(OBJECTS)
	$(AR) crs libtrixiumddr.a $(OBJECTS)

-include $(OBJECTS:.o=.d)

%.o: $(TRIXIUM_BIOS_DIRECTORY)/%.c
	$(compile)

.PHONY: all clean

clean:
	$(RM) $(OBJECTS) libtrixiumddr.a .*~ *~
