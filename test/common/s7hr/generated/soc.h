/* SPDX-License-Identifier: BSD-2-Clause
 *
 * Host test of firmware/ddr_s7hr.c: the YPCB soc.h constants the calibration reads.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef S7HR_FAKE_SOC_H
#define S7HR_FAKE_SOC_H

#define CONFIG_SDRAM_CUSTOM_INIT
#define TRIXIUM_DDR_S7HR
#define TRIXIUM_DDR_CKPHASE_STEPS 112

#endif
