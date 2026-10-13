/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium extra DDR controllers (TRIXIUM_N_MEMCTRL > 1, the XCU1525 and the YPCB): see ddr.c
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef TRIXIUM_DDR_H
#define TRIXIUM_DDR_H

int ddr_init(int k);       // liblitedram's init/leveling on controller k (0 = the BIOS's).
int ddr_init_all(void);    // Controllers 1..N-1.
int ddr_boot_init(void);   // At boot: ddr_init_all() unless the BIOS levels them all.
int ddr_test(int chan);  // Engine STORE/FETCH round trip per channel window (-1 = all).

#endif
