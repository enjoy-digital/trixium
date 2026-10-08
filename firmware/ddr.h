// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// The extra DDR4 controllers (TRIXIUM_N_MEMCTRL > 1, the XCU1525): see ddr.c.
#ifndef TRIXIUM_DDR_H
#define TRIXIUM_DDR_H

int ddr_init(int k);        // liblitedram's sdram_init() on controller k (0 = the BIOS's).
int ddr_init_all(void);     // Controllers 1..N-1, at boot.
int ddr_test(int chan);     // Engine STORE/FETCH round trip per channel window (-1 = all).

#endif
