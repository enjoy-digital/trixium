/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium loader: host-side weight and firmware upload through BAR0
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 * The one upload path trixium_load and trixium_chat share. Words are written through BAR0
 * into the loader's FIFO (CSR path), not the PCIe DMA: LitePCIe's host->device DMA cannot be
 * flow-controlled and silently drops data on underflow.
 *
 * Every function exits the process with a message on a hardware error: the loader only runs
 * interactively or from the deploy scripts.
 */

#ifndef TRIXIUM_LOADER_H
#define TRIXIUM_LOADER_H

#include <stdint.h>

/* Map BAR0 of the devnum'th Xilinx device (sysfs resource0, uncached). Needed before any
 * other call; root. */
void     loader_bar_map(int devnum);

/* Pulse the CPU reset (never the SoC reset: that takes the PCIe endpoint down). The BIOS
 * comes up at its prompt; the weight memory contents survive. */
void     loader_cpu_reset(void);

/* One transfer of `beats` 256-bit beats to (chan, byte address `base`). On bitstreams with
 * TRIXIUM_LOADER_ADDS_CHAN_OFFSET the gateware adds chan*stride itself. */
void     loader_transfer(int chan, uint32_t base, const uint8_t *data, uint64_t beats);

/* Upload a BONS image: header+table to the channel-0 staging area (the firmware's `hdrload`
 * copies it into CPU memory), then every channel's payload to its own window. Returns the
 * number of payload bytes written, 0 on a refused image (channel-count mismatch, not a BONS
 * file). verbose=1 prints the per-channel base lines the deploy scripts grep for. */
uint64_t loader_upload_image(const char *path, int verbose);

/* Copy a firmware binary (linked at MAIN_RAM_BASE) into main_ram through the loader's port:
 * ~1 ms for 100 KB against 30-45 s of 64-byte serial-boot frames. The caller then asks the
 * BIOS to `boot` it. Returns the byte count. */
uint64_t loader_upload_firmware(const char *path);

/* Zero the staged BONS header (no model at boot). */
void     loader_clear_staging(void);

#endif /* TRIXIUM_LOADER_H */
