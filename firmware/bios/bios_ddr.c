/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium BIOS DDR3 calibration (the YPCB's controller 0)
 *
 * liblitedram's sdram_init() calls sdram_custom_init() when the SoC sets
 * CONFIG_SDRAM_CUSTOM_INIT (trixium_ypcb.py): the BIOS then calibrates controller 0 (main_ram)
 * with firmware/ddr_s7hr.c, the same code the firmware runs on controller 1.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include <generated/soc.h>
#include <generated/csr.h>

#if defined(CONFIG_SDRAM_CUSTOM_INIT) && defined(TRIXIUM_DDR_S7HR)
#include "ddr_s7hr.c"
#endif
