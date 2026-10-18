/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium resident FP32 projections and descriptor list builder
 *
 * Resident FP32 projections for the hybrid (27B) path: the activation quantization, the
 * integer Hadamard rotation, the whole-vector requantization and the int32 -> FP32 result
 * restoration run as sequencer descriptor lists inside the scratchpad (units 4, 3, 5 and 8
 * around the engine matvec, resident FETCH/STORE for the FP32 vectors). The CPU builds
 * descriptors and invalidates its cache; it touches no element. Numerically this is the
 * fused CPU path of infer_float.c (same exponent rule, ties-away Q8, int32 K-segment sum, RNE
 * int32 -> FP32 then a power-of-two scale), so the typed-checkpoint gate must stay 87/87.
 *
 * Needs a bitstream with TRIXIUM_HAS_HADAMARD, TRIXIUM_HAS_ACTIVATION_QUANTIZE (ABI 1.1) and
 * TRIXIUM_HAS_PROJECTION_RESTORE. On any other bitstream resident_available() is 0 and
 * hybrid.c keeps its CPU path; a failed list disables the path for the session after an
 * engine soft reset (the weights stay).
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef RESIDENT_H
#define RESIDENT_H

#include <stdint.h>

#include "bons.h"

/* Projections */
/*-------------*/

int  resident_available(void);   // The bitstream carries units 3/4/5/8 with the expected ABIs.
int  resident_init(void);        // Extract the rotation sign bits from the image. 0 = ready.
void resident_enable(int on);    // Console A/B: default on when available.
int  resident_active(void);      // available && enabled && not disabled by a failure.

// Quantize src[width] (FP32, main RAM; NULL = the activation resident_ffn_act left in the
// scratchpad) into the resident Q8 vector, rotating it first when `rotate` (width must be a
// multiple of the Hadamard block). Leaves the whole-vector exponent resident too. 0 = ok, -1 = unsupported/failed (caller falls back).
int  resident_prepare(const float *src, uint32_t width, int rotate);
// Multiply the resident Q8 vector by tensor e (K-segments summed in int32), restore FP32 and
// DMA the rows into dst (main RAM, 32-byte aligned, rows*4 bytes). 0 = ok, -1 = unsupported.
int  resident_matvec(const struct bons_entry *e, float *dst);

/* SiLU Unit */
/*-----------*/

// Unit 11 (TRIXIUM_HAS_SILU): load the firmware's SiLU table into the unit (after
// resident_init), through `stage` (main RAM, 32-byte aligned, 1,025*32 bytes, free after
// the call), and self-test it. 0 = the FFN activation can run resident.
int  resident_silu_init(const float *lut, uint32_t *stage);
int  resident_silu_active(void);
void resident_silu_enable(int on);   // Console A/B.
// With the FFN input prepared (resident_prepare), compute act = up * silu(gate) in the
// scratchpad (bit-exact with hybrid.c's silu_mul_fv; the rare out-of-range elements are
// recomputed on the CPU) and leave it in RS_F for resident_prepare(NULL, ...). act also
// receives a copy in main RAM (the fallback source); gate_out/up_out, when not NULL,
// receive the projections for traces. All 32-byte aligned. 0 = ok, -1 = declined/failed.
int  resident_ffn_act(const struct bons_entry *gate, const struct bons_entry *up,
                      float *act, float *gate_out, float *up_out);
void resident_mark_act(void);      // RS_F holds the next prepare's FP32 source (a list left it)
// The next resident_matvec into dst leaves its result in RS_F only - no STORE, no cache
// maintenance - for a caller whose next list reads it there (rl_rsf_is): dst's main-RAM copy
// is then stale. The next rl_begin or resident_ffn_act forgets the request.
void resident_keep_rsf(const void *dst);

/* List Builder */
/*--------------*/

// List builder: compose one descriptor list from transfers, prepares, matvecs
// and raw descriptors, then run it. Any invalid step marks the list bad: rl_run returns -1
// without running anything. Fixed regions shared with the calls above:
#define RS_Q8B_ADDR 0x00000u
#define RS_Y_ADDR   0x04800u
#define RS_F_ADDR   0x1C800u
struct bns_seq_descriptor;
void rl_begin(void);
// rl_begin naming the list (the list image cache, resident.c): a, b, c are everything its
// descriptors depend on (a: a site code in bits 31:24 and the rest - never 0); returns 1 when
// the list is a replay of an earlier build, so the caller may skip its own encoding work (the
// rl_* calls must still be made, in the same order: they keep the count and the cache
// maintenance).
int  rl_begin_keyed(uint32_t a, uint32_t b, uint32_t c);
int  rl_replaying(void);
void rl_fetch(const void *src, uint32_t bytes, uint32_t sp);
void rl_store(void *dst, uint32_t bytes, uint32_t sp);
// No cache maintenance, for main-RAM spans the CPU holds no dirty line of (constants cleaned
// once, device-owned buffers): rl_fetch cleans its source and rl_store flushes and then
// invalidates its destination, ~5 cycles per line (significant on the DeltaNet conv's
// 320 KB per layer).
void rl_fetch_raw(const void *src, uint32_t bytes, uint32_t sp);
void rl_store_raw(void *dst, uint32_t bytes, uint32_t sp);
void rl_matvec(const struct bons_entry *e, uint32_t ystage, uint32_t dst_sp);
// Rows [row0, row0 + rows) of each channel's block of e, row0 and rows multiples of 8*n_ch:
// the image shards rows in blocks (channel c holds model rows [c*R/n_ch, (c+1)*R/n_ch)), so
// channel c computes its local rows row0/n_ch .. and the result is channel-major - model
// rows c*R/n_ch + row0/n_ch + j at slot c*rows/n_ch + j. Contiguous in the model only when
// the call covers the whole tensor; otherwise store it with rl_store_packets.
void rl_matvec_rows(const struct bons_entry *e, uint32_t row0, uint32_t rows, uint32_t ystage,
                    uint32_t dst_sp);
// NOWAIT on the last descriptor added (an engine op: the next unit descriptors added with
// rl_desc_overlap run under it), and rl_desc with OVERLAP. The caller asserts independence.
void rl_nowait(void);
void rl_desc_overlap(const struct bns_seq_descriptor *d);
// One engine STORE of `packets` runs of packet_bytes: run r from sp + r*sp_stride to
// dst + r*dst_stride (bytes, 32-byte multiples; <= 31 runs, sp_stride <= 8160). inv/inv_bytes:
// the main-RAM span to clean now and invalidate after the list (NULL: the caller does it).
void rl_store_packets(void *dst, uint32_t packet_bytes, uint32_t packets, uint32_t dst_stride,
                      uint32_t sp, uint32_t sp_stride, void *inv, uint32_t inv_bytes);
void rl_prepare(const float *src, uint32_t width, int rotate);   // src NULL: already in RS_F
void rl_desc(const struct bns_seq_descriptor *d);
struct engine_op;
void rl_engine(const struct engine_op *op);                       // any engine op, in order
// The attention output gate, dst = m * sigmoid(g) on unit 11 (table only): rl_gate_begin loads
// the sigmoid table through the 1,025-beat image area img_sp, rl_gate_seg adds one run (dst
// disjoint from g and m, or == g; fixmap 32-byte aligned), rl_gate_end reloads the SiLU table.
// After the list, resident_gate_patch() recomputes each run's flagged elements from the
// scratchpad's g and m into dst (main RAM).
int  resident_gate_init(const float *sig_lut, uint32_t *image);
int  resident_gate_active(void);
void rl_gate_begin(uint32_t img_sp);
void rl_gate_seg(uint32_t g_sp, uint32_t m_sp, uint32_t dst_sp, uint32_t fixmap_sp, uint32_t n);
void rl_gate_end(uint32_t img_sp);
int  resident_silu_restore(void);    // reload the SiLU table (after a failed gated list)
uint32_t resident_gate_patch(uint32_t fixmap_sp, uint32_t g_sp, uint32_t m_sp, float *dst,
                             uint32_t n);
int  rl_run(const char *what);
// rl_run in two halves: the CPU may work between them on anything but the list's scratchpad
// regions and main-RAM buffers and the engine - including building the next list (the slots
// have two banks). A failed start needs no rl_wait.
int  rl_start(const char *what);
int  rl_wait(void);
// rl_start without a wait: the list stays in flight while the caller builds the next one,
// whose start (rl_start, rl_run, any resident_* list) waits for it first; resident_join()
// waits for it explicitly (before the CPU reads what it wrote, or uses the engine). Its
// failure is reported by that start or join. resident_prepare, and resident_matvec with
// resident_keep_rsf, post their lists the same way.
int  rl_post(const char *what);
int  resident_join(void);
// Descriptors built so far (an index for rl_wait_desc), and a wait inside a started list
// until the first k have completed: the CPU may then read their scratchpad outputs.
uint32_t rl_count(void);
void resident_rsf_set(const void *src);   // RS_F now holds src (FP32), after a list
int  rl_rsf_is(const void *src);          // the list being built still has src in RS_F
int  rl_wait_desc(uint32_t k);
// Unit 11 fixups for such lists (m_sp 0: m = 1.0; perm_a > 1: the unit's permuted output).
uint32_t resident_silu_patch(uint32_t g_sp, uint32_t m_sp, uint32_t dst_sp, uint32_t fixmap_sp,
                             uint32_t n, uint32_t seg, uint32_t perm_a, uint32_t perm_b);

/* State */
/*-------*/

// State and per-forward accounting (cycles include the descriptor build and the CPU-side
// cache work). Lives in the main_ram static window (static_memory.h), not SRAM .bss: the
// 8 KiB SRAM is at its 4 KiB stack floor. Initialized by resident_init(); the flags first.
struct resident_state {
    uint64_t prepare_cycles, matvec_cycles;
    uint32_t prepares, matvecs, fallbacks;
    uint32_t sign_base[4];       // Scratchpad offset of each Hadamard width's sign bits, 0 = none.
    const float *prep_src;       // hybrid.c: the last prepared source, for a CPU fallback.
    int32_t prep_width, prep_rotate;
    uint8_t enabled, disabled, ready, n_ch;
    uint64_t silu_cycles;
    uint32_t silu_lists, silu_fixups;
    const float *silu_lut;
    uint8_t silu_enabled, silu_ready, act_resident;
    // The FP32 vector RS_F holds after the last list (resident_rsf_set; every list start
    // forgets it), and the same for the list being built until a descriptor may write RS_F:
    // a prepare of that vector skips its FETCH (~2.4k cycles for a 27B width).
    const void *rsf_src, *rsf_list;
    // Attention output gate on unit 11 (table only, ABI >= 1.2): both table images stay in
    // main RAM and a gated list swaps the sigmoid's in and the SiLU's back.
    const uint32_t *silu_image, *sig_image;
    const float *sig_lut;
    uint8_t gate_ready, gate_set;   // gate_set: the sigmoid is resident in table set 1 (ABI 1.3)
    uint32_t gate_fixups;
    const void *keep_dst;           // resident_keep_rsf
    uint8_t half;                   // The image's block is half the unit's (512 on 1,024).
};
struct resident_state *resident_state(void);
void resident_stats_reset(void);
void resident_listprof_print(uint32_t forwards);   // `listprof` lines (stats)
void resident_listprof_names(void);                // `TPL` lines (tprof dump)
void resident_listcache_mode(int mode);            // 0 off, 1 on, 2 verify (every image rebuilt)
void resident_listcache_print(void);               // `listcache` line (stats)
void resident_reenable(void);                      // the path back on after a list failed on corrupt reads

#endif /* RESIDENT_H */
