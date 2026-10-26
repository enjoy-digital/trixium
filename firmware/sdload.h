/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium weights from an SD card / eMMC
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef TRIXIUM_SDLOAD_H
#define TRIXIUM_SDLOAD_H

#include <stdint.h>

// The device commands; the slots and tables (sdboot/sdsave/sdls/sddefault) are storage.h's,
// shared with the SATA path.

// tries: CMD8 attempts (each waits out the command timeout without a card).
int sd_init(int mhz, int tries);
int sd_init_done(void);
int sd_load(int chan, uint32_t base, uint32_t sector, uint32_t blocks);
// n blocks to the card from a DRAM window (chan, base), or from CPU memory when cpu != NULL.
int sd_write(uint32_t sector, uint32_t blocks, int chan, uint32_t base, const uint32_t *cpu);

#endif
