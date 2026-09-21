/* Host-side HBM loader: the one upload path both bonsai_load and bonsai_chat use.
 *
 * Words are poked through BAR0 into the loader's FIFO (CSR path). The DMA path was retired:
 * litepcie's host->device DMA cannot be flow-controlled and silently dropped the first ring
 * (the staged BONS header and the first megabyte of channel 0) on every upload - see
 * docs/bringup.md "Weight upload". bonsai_chat -m kept using it long after bonsai_load was
 * fixed (2026-09-11), which is why this is now one module.
 *
 * Every function exits the process on a hardware error with a message; the loader is only
 * ever run interactively or from the deploy scripts.
 */
#ifndef BONSAI_LOADER_H
#define BONSAI_LOADER_H

#include <stdint.h>

/* Map BAR0 of the devnum'th Xilinx device (sysfs resource0, uncached). Needed before any
 * other call; root. */
void     loader_bar_map(int devnum);

/* Pulse the CPU reset (never the SoC reset: that takes the PCIe endpoint down). The BIOS
 * comes up at its prompt; HBM contents survive. */
void     loader_cpu_reset(void);

/* One transfer of `beats` 256-bit beats to HBM at (chan, byte address `base`). On bitstreams
 * with BONSAI_LOADER_ADDS_CHAN_OFFSET the gateware adds chan*stride itself. */
void     loader_transfer(int chan, uint32_t base, const uint8_t *data, uint64_t beats);

/* Experimental bounded region upload (zzregion ABI), not legacy channel placement.
 * Requires matching native-build headers, exclusive device ownership and a
 * prevalidated model allocation. bytes/offset must be 32-byte aligned; padding
 * belongs to that allocation. Does not reset on failure or verify by readback.
 */
void     loader_transfer_region(uint32_t region, uint64_t offset, const uint8_t *data, uint64_t bytes);

/* Upload a BONS image: header+table to the channel-0 staging area (the firmware's
 * `hdrload` copies it into CPU memory), then every channel's payload to its own window.
 * Returns the number of payload bytes written, 0 on a refused image (channel-count
 * mismatch, not a BONS file). verbose=1 prints the per-channel base lines the deploy
 * scripts grep for. */
uint64_t loader_upload_image(const char *path, int verbose);

/* Copy a firmware binary (linked at MAIN_RAM_BASE) into main_ram through the loader - HBM
 * pseudo-channel 0 at offset 0, reached through the loader's port because the HBM AXI
 * switch routes by address. ~1 ms for 100 KB against 30-45 s of 64-byte serial-boot frames.
 * The caller then asks the BIOS to `boot 0x40000000`. Returns the byte count. */
uint64_t loader_upload_firmware(const char *path);
void     loader_clear_staging(void);   /* Zero the staged BONS header (no model at boot). */

#endif
