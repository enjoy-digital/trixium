/* SPDX-License-Identifier: BSD-2-Clause
 *
 * Host test of firmware/ddr_s7hr.c: the generated sdram_phy.h of a YPCB controller (the
 * S7HRDDRPHY's settings), its DFII helpers on the fake board.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef S7HR_FAKE_SDRAM_PHY_H
#define S7HR_FAKE_SDRAM_PHY_H

#include <generated/csr.h>

#define DFII_CONTROL_SEL     0x01
#define DFII_CONTROL_CKE     0x02
#define DFII_CONTROL_ODT     0x04
#define DFII_CONTROL_RESET_N 0x08

#define DFII_COMMAND_CS     0x01
#define DFII_COMMAND_WE     0x02
#define DFII_COMMAND_CAS    0x04
#define DFII_COMMAND_RAS    0x08
#define DFII_COMMAND_WRDATA 0x10
#define DFII_COMMAND_RDDATA 0x20

#define SDRAM_PHY_S7HRDDRPHY
#define SDRAM_PHY_DATABITS 64
#define SDRAM_PHY_DFI_DATABITS 128
#define SDRAM_PHY_PHASES 4
#define SDRAM_PHY_RDPHASE 2
#define SDRAM_PHY_WRPHASE 3
#define SDRAM_PHY_MODULES 8
#define SDRAM_PHY_DELAYS 32
#define SDRAM_PHY_BITSLIPS 8
#define SDRAM_PHY_DDR3

#define DFII_PIX_DATA_SIZE 4

void cdelay(int i);
void fake_init_sequence(void);

static inline void command_p0(int cmd) { sdram_dfii_pi_command_issue(0, cmd); }
static inline void command_p1(int cmd) { sdram_dfii_pi_command_issue(1, cmd); }
static inline void command_p2(int cmd) { sdram_dfii_pi_command_issue(2, cmd); }
static inline void command_p3(int cmd) { sdram_dfii_pi_command_issue(3, cmd); }

static inline unsigned long sdram_dfii_pix_wrdata_addr(int phase) { return FAKE_WRDATA_BASE + 0x10*phase; }
static inline unsigned long sdram_dfii_pix_rddata_addr(int phase) { return FAKE_RDDATA_BASE + 0x10*phase; }

static inline void init_sequence(void) { fake_init_sequence(); }

#endif
