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
int ddr_test(int chan);
uint32_t ddr_test_deep(int blocks, int chan); // Destructive deep test: failing controllers mask.
void ddr_tap_offset(int k, int off);
int ddr_bit_offset(int k, int dq, int off); // One DQ bit's read tap = leveled + off.
int ddr_ecc_scan(int k, int range, int passes); // Every bit's read tap to the centre of its eye under BIST traffic (ECC per-bit counters): bits moved, -1 = not the firmware's leveling.
void ddr_mode_register(int k, int mr, int value); // Diagnostic: MRS on controller k > 0. // Diagnostic: last leveling's read taps + off.  // Engine STORE/FETCH round trip per channel window (-1 = all).

#endif
