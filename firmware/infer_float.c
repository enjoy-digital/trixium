/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium float matvec primitives of the hybrid path
 *
 * The engine matvec on float vectors (quantized to int8 with a power-of-two scale), its
 * K-segment sums, the fused Prism rotation, and the CPU embedding and RoPE.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <generated/csr.h>
#include <generated/soc.h>
#include <generated/mem.h>

#include <system.h>

#include "bons.h"
#include "engine.h"
#include "hybrid.h"
#include "model.h"
#include "infer.h"
#include "static_memory.h"
#include "ops.h"
#include "perf.h"
#include "vec.h"
#include "seq.h"
#include "progs.h"
#include "infer_internal.h"

/* Float Matvec */
/*--------------*/

// Hybrid float-result staging.  CPU reads of the scratchpad use a registered 32-bit
// Wishbone sideband (several round trips per word).  On the XCU1525 the engine can STORE
// the gathered SP_Y stream into controller-0 main RAM, where the CPU consumes it through
// its burst-filled D-cache.  This buffer is read-only to the CPU, so post-DMA invalidation
// cannot discard dirty data.  Targets without STORE + Zicbom keep the direct SP path.
static int32_t *i32_stage;

static void matvec_engine_rows(const struct bons_entry *e, uint32_t row0, uint32_t rows,
                               uint32_t sp_dst)
{
    uint32_t rb = bons_row_beats(e->cols, e->kind, e->group);
    struct engine_op op = {0};
    op.kind   = (e->kind == KIND_Q8) ? ENGINE_KIND_Q8 : ENGINE_KIND_TERNARY;
    op.rows   = rows;
    op.cols   = e->cols;
    op.wbase  = TRIXIUM_CHAN_WEIGHT_BASE + (e->chan_base_beats + (uint64_t)row0*rb)*32;
    op.wlen   = rows*rb;
    op.sbeats = bons_scale_beats(e->cols, e->group);
    op.cbeats = bons_code_beats(e->cols, e->kind);
    op.bias   = e->bias;
    op.rshift = e->rshift;
    op.xsrc   = SP_X;
    op.dst    = sp_dst;
    engine_run(&op);
}

static const volatile int32_t *stage_i32_results_to(uint32_t sp_src, uint32_t count,
                                                     int dma_result, int32_t *stage)
{
    const volatile int32_t *y = (volatile int32_t *)(SCRATCHPAD_BASE + sp_src);
#if defined(HAS_INVD_CPU_DCACHE_RANGE)
    if (dma_result) {
        uint32_t eng_addr = (uint32_t)((uintptr_t)stage - MAIN_RAM_BASE
                                      + TRIXIUM_MEM_CPU_OFFSET);
        engine_store(0, eng_addr, (count*4 + 31)/32, sp_src);
        invd_cpu_dcache_range(stage, count*4);
        y = stage;
    }
#else
    (void)dma_result;
    (void)stage;
#endif
    return y;
}

static const volatile int32_t *stage_i32_results(uint32_t sp_src, uint32_t count,
                                                  int dma_result)
{
    return stage_i32_results_to(sp_src, count, dma_result, i32_stage);
}

// Run an engine matvec for tensor e on rows [row0, row0+rows) per channel; results (float,
// scaled by xscale) into dst[gathered global order: chan*rows + row]. With `acc` the int32
// results are added into acc[] instead (a K-segment of a wide matvec; the caller scales).
COLD void matvec_rows_f(const struct bons_entry *e, uint32_t row0, uint32_t rows,
                               float xscale, float *dst, int32_t *acc, int acc_add, int dma_result)
{
    matvec_engine_rows(e, row0, rows, SP_Y);

    perf_enter(OP_NORM_I32);
    uint32_t count = rows*n_ch;
    const volatile int32_t *y = stage_i32_results(SP_Y, count, dma_result);
    if (acc) {
        if (acc != (const int32_t *)y || acc_add) {
            uint32_t i = 0;
            if (acc_add) {
                for (; i + 3 < count; i += 4) {
                    acc[i + 0] += y[i + 0]; acc[i + 1] += y[i + 1];
                    acc[i + 2] += y[i + 2]; acc[i + 3] += y[i + 3];
                }
                for (; i < count; i++) acc[i] += y[i];
            } else {
                for (; i + 3 < count; i += 4) {
                    acc[i + 0] = y[i + 0]; acc[i + 1] = y[i + 1];
                    acc[i + 2] = y[i + 2]; acc[i + 3] = y[i + 3];
                }
                for (; i < count; i++) acc[i] = y[i];
            }
        }
    } else {
        float sc = ldexpf(xscale, (int)e->rshift - (int)e->bias);
        uint32_t i = 0;
        // Four independent conversions let the in-order RV32F core overlap fcvt/fmul
        // latency and remove three quarters of the loop-control instructions.
        for (; i + 3 < count; i += 4) {
            float f0 = (float)y[i + 0], f1 = (float)y[i + 1];
            float f2 = (float)y[i + 2], f3 = (float)y[i + 3];
            dst[i + 0] = f0*sc; dst[i + 1] = f1*sc;
            dst[i + 2] = f2*sc; dst[i + 3] = f3*sc;
        }
        for (; i < count; i++) dst[i] = (float)y[i]*sc;
    }
    perf_exit(OP_NORM_I32);
}

// Full matvec (all rows), output in original row order (block shard: chan-major == row order).
// The float matvec the hybrid CPU path runs (hybrid.c): quantize the input to int8 with a
// power-of-two scale, engine, float results in row order - chunked so the int32 results
// stay inside SP_Y..SP_H (96 KB) whatever the row count (the 27B's LM head is 15,520 rows
// per channel).
static float *rot_scratch;   // (SP_H - SP_Y)/4 floats/ints, allocated on first use.

static void ensure_float_matvec_scratch(void)
{
    if (!rot_scratch) {
        rot_scratch = halloc((SP_H - SP_Y));
        i32_stage   = halloc((SP_H - SP_Y));
    }
}

static void matvec_f_one(const struct bons_entry *e, float xs, float *dst, int32_t *acc,
                         int acc_add)
{
    uint32_t rpc   = e->rows/n_ch;
    uint32_t chunk = (SP_H - SP_Y)/(4*n_ch);
    if (chunk > rpc) chunk = rpc;
    for (uint32_t r0 = 0; r0 < rpc; r0 += chunk) {
        uint32_t n = (rpc - r0 < chunk) ? rpc - r0 : chunk;
        // matvec_rows_f gathers chan*n + row: unshuffle into row order per channel.
        if (n == rpc) {
            matvec_rows_f(e, r0, n, xs, dst, acc, acc_add, 1);
        } else if (acc) {
            // K-segment: the int32 results arrive as [chan][n]; add them at [chan][rpc].
            matvec_rows_f(e, r0, n, xs, 0, (int32_t *)rot_scratch, 0, 1);
            for (int c = 0; c < n_ch; c++)
                for (uint32_t i = 0; i < n; i++)
                    if (acc_add)
                        acc[(uint32_t)c*rpc + r0 + i] += ((int32_t *)rot_scratch)[(uint32_t)c*n + i];
                    else
                        acc[(uint32_t)c*rpc + r0 + i] = ((int32_t *)rot_scratch)[(uint32_t)c*n + i];
        } else {
            // Results arrive as [chan][n] in rot_scratch; scatter them into the caller's
            // [chan][rpc] row order.
            matvec_rows_f(e, r0, n, xs, rot_scratch, 0, 0, 1);
            for (int c = 0; c < n_ch; c++)
                for (uint32_t i = 0; i < n; i++)
                    dst[(uint32_t)c*rpc + r0 + i] = rot_scratch[(uint32_t)c*n + i];
        }
    }
}

// K-segments (FLAG_KSEG, the 27B's FFN down projection): the whole input is quantised once
// with one exponent, each segment's int8 slice goes to SP_X, the engine's int32 results add
// up in acc[] (every segment carries the tensor's bias/rshift, so they are in one fixed
// point) and are scaled once - the unsplit result up to one LSB of the per-segment shift.
uint32_t infer_matvec_cols(const struct bons_entry *e)   // K of the whole matvec.
{
    if (!(e->flags & FLAG_KSEG))
        return e->cols;
    uint32_t cols = 0;
    int nseg = model_entry_nkseg(e);   // once, not in the loop condition
    for (int k = 0; k < nseg; k++)
        cols += model_entry_kseg(e, k)->cols;
    return cols;
}

static int32_t *kseg_acc;    // rows int32, sized on first use to the widest such tensor.

int infer_quantize_f(const float *src, uint32_t n)
{
    return quantize_f_pow2(src, n, SP_X);
}

/* Fused Prism Rotation */
/*----------------------*/

// Rather than materialising every Hadamard block as FP32 in main RAM, scanning the
// concatenated vector for its Q8 exponent, then reading it all again to feed the immediately
// following projection: a rotation result is an int32 engine value times a power of two, so
// retain those integers in the cacheable DMA buffer and remember the exponent per block.
// The global exponent comes from the engine's exact absmax, and packing combines the two
// power-of-two scales around the same RV32F conversion.  Thus the Q8 bytes are identical,
// without the intermediate float write, float scan and float reload.
#define ROT_BLOCK_MAX 64
static int rot_block_e[ROT_BLOCK_MAX];
static int rot_stage_width, rot_stage_block, rot_stage_e8;

static void pack_rotated_range(uint32_t c0, uint32_t n, uint32_t sp_dst)
{
    const int32_t *raw = (const int32_t *)rot_scratch;
    volatile uint32_t *x = (volatile uint32_t *)(SCRATCHPAD_BASE + sp_dst);
    uint32_t end = c0 + n, out = 0;
    for (uint32_t i = c0; i < end;) {
        uint32_t b = i/(uint32_t)rot_stage_block;
        uint32_t bend = (b + 1u)*(uint32_t)rot_stage_block;
        if (bend > end) bend = end;
        float is = ldexpf(1.0f, rot_block_e[b] - rot_stage_e8);
        for (; i < bend; i += 4)
            x[out++] = i32_pack_q8x4(raw + i, is);
    }
}

int infer_rotate_quantize_f(const float *src, uint32_t width)
{
#if !defined(HAS_INVD_CPU_DCACHE_RANGE)
    (void)src;
    (void)width;
    return INFER_ROTATE_UNSUPPORTED;
#else
    ensure_float_matvec_scratch();
    int B = H->hadamard_block, wi = -1;
    for (int i = 0; i < 4; i++)
        if (H->hadamard_widths[i] == width) wi = i;
    if (!B || wi < 0 || width % (uint32_t)B || B % n_ch ||
        width/(uint32_t)B > ROT_BLOCK_MAX || width*4u > (SP_H - SP_Y)) {
        printf("hybrid: unsupported fused rotation width %lu block %d\n",
               (unsigned long)width, B);
        return INFER_ROTATE_UNSUPPORTED;
    }

    rot_stage_width = (int)width;
    rot_stage_block = B;
    uint32_t max_bits = 0;
    int32_t *raw = (int32_t *)rot_scratch;
    for (uint32_t b = 0; b < width/(uint32_t)B; b++) {
        const struct bons_entry *re = model_entry(ROLE_HADAMARD, HADAMARD_LAYER(wi, b));
        int xe = quantize_f_pow2(src + b*(uint32_t)B, B, SP_X);
        matvec_engine_rows(re, 0, B/n_ch, SP_Y);
#ifdef CSR_TRIXIUM_ENGINE_CSR_ABSMAX_ADDR
        uint32_t amax = trixium_engine_csr_absmax_read();
#else
        uint32_t amax = 0;
#endif
        perf_enter(OP_NORM_I32);
        const volatile int32_t *y = stage_i32_results_to(
            SP_Y, B, 1, raw + b*(uint32_t)B);
#ifndef CSR_TRIXIUM_ENGINE_CSR_ABSMAX_ADDR
        for (int i = 0; i < B; i++) {
            int64_t v = y[i];
            uint32_t a = (uint32_t)(v < 0 ? -v : v);
            if (a > amax) amax = a;
        }
#else
        (void)y;
#endif
        perf_exit(OP_NORM_I32);
        int oe = xe + (int)re->rshift - (int)re->bias;
        rot_block_e[b] = oe;
        float mf = (float)amax*ldexpf(1.0f, oe);
        uint32_t bits = f32_abs_bits(mf);
        if (bits <= 0x7f800000u && bits > max_bits) max_bits = bits;
    }
    rot_stage_e8 = f32_e8_from_amax_bits(max_bits);
    if (width <= SP_Y - SP_X) {
        perf_enter(OP_QUANT8);
        pack_rotated_range(0, width, SP_X);
        perf_exit(OP_QUANT8);
    }
    return rot_stage_e8;
#endif
}

void infer_matvec_f_rotated_quantized(const struct bons_entry *e, int e8, float *dst)
{
    ensure_float_matvec_scratch();
    uint32_t cols = infer_matvec_cols(e);
    if (cols <= SP_Y - SP_X) {
        infer_matvec_f_quantized(e, e8, dst);
        return;
    }
    if (!(e->flags & FLAG_KSEG) || cols != (uint32_t)rot_stage_width ||
        e8 != rot_stage_e8 || (uint64_t)e->rows*4 > (SP_H - SP_YU)) {
        printf("hybrid: unsupported fused wide projection K%lu R%lu\n",
               (unsigned long)cols, (unsigned long)e->rows);
        return;
    }

    int nseg = model_entry_nkseg(e);
    uint32_t c0 = 0;
    for (int k = 0; k < nseg; k++) {
        const struct bons_entry *seg = model_entry_kseg(e, k);
        perf_enter(OP_QUANT8);
        pack_rotated_range(c0, seg->cols, SP_X);
        perf_exit(OP_QUANT8);
        matvec_engine_rows(seg, 0, seg->rows/n_ch, k ? SP_YU : SP_Y);
        if (k) {
            perf_enter(OP_NORM_I32);
            int32_t a[4] = {SP_Y, SP_YU, SP_Y, (int32_t)((e->rows + 7)/8)};
            vec_run(PROG_REDUCE_ACC, a, 4);
            perf_exit(OP_NORM_I32);
        }
        c0 += seg->cols;
    }
    perf_enter(OP_NORM_I32);
    const volatile int32_t *y = stage_i32_results(SP_Y, e->rows, 1);
    float sc = ldexpf(1.0f, e8 + (int)e->rshift - (int)e->bias);
    uint32_t i = 0;
    for (; i + 3 < e->rows; i += 4) {
        float f0 = (float)y[i + 0], f1 = (float)y[i + 1];
        float f2 = (float)y[i + 2], f3 = (float)y[i + 3];
        dst[i + 0] = f0*sc; dst[i + 1] = f1*sc;
        dst[i + 2] = f2*sc; dst[i + 3] = f3*sc;
    }
    for (; i < e->rows; i++) dst[i] = (float)y[i]*sc;
    perf_exit(OP_NORM_I32);
}

/* Hybrid Path Primitives */
/*------------------------*/

void infer_matvec_f_quantized(const struct bons_entry *e, int e8, float *dst)
{
    // SP_X is read-only to the engine, so consecutive non-K-segment projections with the
    // same input can reuse its exact bytes and exponent.  hybrid.c is the only caller.
    ensure_float_matvec_scratch();
    matvec_f_one(e, ldexpf(1.0f, e8), dst, 0, 0);
}

void infer_matvec_f(const struct bons_entry *e, const float *src, float *dst)
{
    ensure_float_matvec_scratch();
    if (!(e->flags & FLAG_KSEG)) {
        int e8 = quantize_f_pow2(src, e->cols, SP_X);
        matvec_f_one(e, ldexpf(1.0f, e8), dst, 0, 0);
        return;
    }
    int nseg = model_entry_nkseg(e);
    uint32_t cols = infer_matvec_cols(e);
    if (!kseg_acc) {
        // The widest K-split tensor decides both buffers (the table is small; once).
        uint32_t max_rows = 0;
        const struct bons_entry *tab = model_entry_at(0);
        for (uint32_t i = 0; i < H->n_tensors; i++) {
            const struct bons_entry *t = &tab[i];
            if (!(t->flags & FLAG_KSEG) || t->kseg != 0) continue;
            if (t->rows > max_rows) max_rows = t->rows;
        }
        kseg_acc = halloc(max_rows*4);
    }
    // One exponent for the whole vector (quantize_f_pow2's rule: amax's exponent - 7).
    int e8 = f32_quant_e8(src, cols);
    float is = ldexpf(1.0f, -e8);
    uint32_t c0 = 0;
    // The 27B's two FFN-down segments each produce one 20 KiB int32 vector.  Keep both
    // inside the two scratchpad result banks, add later segments into the first with the
    // existing 8-lane vector program, then DMA/convert only the final sum.  This removes
    // one controller-0 STORE plus both CPU-side accumulator passes per layer.  Wider result
    // tensors retain the generic main-RAM accumulator below.
    if ((uint64_t)e->rows*4 <= (SP_H - SP_YU)) {
        for (int k = 0; k < nseg; k++) {
            const struct bons_entry *seg = model_entry_kseg(e, k);
            perf_enter(OP_QUANT8);
            volatile uint32_t *x = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_X);
            for (uint32_t i = 0; i < seg->cols; i += 4)
                x[i/4] = f32_pack_q8x4(src + c0 + i, is);
            perf_exit(OP_QUANT8);
            matvec_engine_rows(seg, 0, seg->rows/n_ch, k ? SP_YU : SP_Y);
            if (k) {
                perf_enter(OP_NORM_I32);
                int32_t a[4] = {SP_Y, SP_YU, SP_Y, (int32_t)((e->rows + 7)/8)};
                vec_run(PROG_REDUCE_ACC, a, 4);
                perf_exit(OP_NORM_I32);
            }
            c0 += seg->cols;
        }
        perf_enter(OP_NORM_I32);
        const volatile int32_t *y = stage_i32_results(SP_Y, e->rows, 1);
        float sc = ldexpf(1.0f, e8 + (int)e->rshift - (int)e->bias);
        uint32_t i = 0;
        for (; i + 3 < e->rows; i += 4) {
            float f0 = (float)y[i + 0], f1 = (float)y[i + 1];
            float f2 = (float)y[i + 2], f3 = (float)y[i + 3];
            dst[i + 0] = f0*sc; dst[i + 1] = f1*sc;
            dst[i + 2] = f2*sc; dst[i + 3] = f3*sc;
        }
        for (; i < e->rows; i++) dst[i] = (float)y[i]*sc;
        perf_exit(OP_NORM_I32);
        return;
    }
    for (int k = 0; k < nseg; k++) {
        const struct bons_entry *seg = model_entry_kseg(e, k);
        // The shared exponent is what makes quantisation global; quantising each slice
        // directly to SP_X produces the same bytes as quantising the whole vector first,
        // without the extra memory passes.
        perf_enter(OP_QUANT8);
        volatile uint32_t *x = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_X);
        for (uint32_t i = 0; i < seg->cols; i += 4)
            x[i/4] = f32_pack_q8x4(src + c0 + i, is);
        perf_exit(OP_QUANT8);
        matvec_f_one(seg, 1.0f, 0, kseg_acc, k != 0);
        c0 += seg->cols;
    }
    float sc = ldexpf(1.0f, e8 + (int)e->rshift - (int)e->bias);
    uint32_t i = 0;
    for (; i + 3 < e->rows; i += 4) {
        float f0 = (float)kseg_acc[i + 0], f1 = (float)kseg_acc[i + 1];
        float f2 = (float)kseg_acc[i + 2], f3 = (float)kseg_acc[i + 3];
        dst[i + 0] = f0*sc; dst[i + 1] = f1*sc;
        dst[i + 2] = f2*sc; dst[i + 3] = f3*sc;
    }
    for (; i < e->rows; i++) dst[i] = (float)kseg_acc[i]*sc;
}

void *infer_halloc(uint64_t n, int big)
{
    uint8_t *lim = big && heap_hi ? (uint8_t *)(0x40000000ull + MAIN_RAM_SIZE - (2u << 20))
                                  : (uint8_t *)(0x40000000ull + MAIN_RAM_SIZE - (1u << 20));
    uint8_t *p = big && heap_hi ? heap_hi : heap;
    if ((uint64_t)(lim - p) < n) return NULL;
    return big && heap_hi ? halloc_hi((uint32_t)n) : halloc((uint32_t)n);
}

uint64_t infer_heap_room(int big)
{
    uint8_t *lim = (uint8_t *)(0x40000000ull + MAIN_RAM_SIZE - (1u << 20));
#if !defined(TRIXIUM_CHAN_MOD) || TRIXIUM_CHAN_MOD == 0
    // One flat memory (the FK33): the low heap ends where the first engine window inside
    // main_ram starts (channel 3's, at +64 MB) - hybrid models are not this board's.
    if (MAIN_RAM_SIZE > (64u << 20)) lim = (uint8_t *)(0x40000000u + (64u << 20));
#endif
    uint8_t *p = big && heap_hi ? heap_hi : heap;
    if (big && heap_hi) lim = (uint8_t *)(0x40000000ull + MAIN_RAM_SIZE - (2u << 20));
    return p < lim ? (uint64_t)(lim - p) : 0;
}

void infer_fetch_f32(uint32_t addr, uint32_t beats, float *dst, int n)
{
#if defined(HAS_INVD_CPU_DCACHE_RANGE)
    // Whole beats into a 32-byte aligned destination (every halloc'd table): the engine
    // stores each staged 8 KB back to main_ram instead of the CPU reading it word by word
    // over the uncached bus (~30 cycles a float; the hybrid models' norm and conv tables are
    // 19k floats on the tiny model, millions on the 27B). Same bytes: the lines are written
    // back before the STOREs and dropped after them.
    if (((uintptr_t)dst & 31) == 0 && n > 0 && n % 8 == 0 && (uint32_t)n/8 <= beats) {
        flush_cpu_dcache_range(dst, (size_t)n*4);
        for (int done = 0; done < n; ) {
            uint32_t b = (uint32_t)(n - done)/8;
            if (b > 256) b = 256;
            engine_fetch(0, addr + (uint32_t)done*4, b, SP_FET);
            engine_store(0, (uint32_t)((uintptr_t)(dst + done) - MAIN_RAM_BASE
                                       + TRIXIUM_MEM_CPU_OFFSET), b, SP_FET);
            done += (int)b*8;
        }
        invd_cpu_dcache_range(dst, (size_t)n*4);
        return;
    }
#endif
    // Through the fetch staging region, 8 KB at a time.
    int done = 0;
    while (done < n) {
        uint32_t b = beats - (uint32_t)done/8;
        if (b > 256) b = 256;
        engine_fetch(0, addr + (uint32_t)done*4, b, SP_FET);
        volatile float *p = (volatile float *)(SCRATCHPAD_BASE + SP_FET);
        int m = (int)b*8;
        if (m > n - done) m = n - done;
        for (int i = 0; i < m; i++) dst[done + i] = p[i];
        done += m;
    }
}

/* CPU Embedding and RoPE */
/*------------------------*/

// Embedding row: fetch the row packet from HBM and dequantize (ternary).
COLD void embed_row_f(const struct bons_entry *e, int row, float *dst)
{
    uint32_t rb  = bons_row_beats(e->cols, e->kind, e->group);
    uint32_t rpc = e->rows/n_ch; // Rows per channel.
    uint32_t ch  = row/rpc;
    uint32_t loc = row % rpc;
    engine_fetch(ch, TRIXIUM_CHAN_WEIGHT_BASE + (e->chan_base_beats + (uint64_t)loc*rb)*32,
                 rb, SP_FET);
    volatile uint8_t *p = (volatile uint8_t *)(SCRATCHPAD_BASE + SP_FET);
    uint32_t sb = bons_scale_beats(e->cols, e->group);
    const volatile uint16_t *scales = (const volatile uint16_t *)p;
    const volatile uint8_t  *codes  = p + sb*32;
    for (int g = 0; g < (int)(e->cols/128); g++) {
        float s = fp16_to_float(scales[g]);
        for (int j = 0; j < 128; j++) {
            int q8 = (codes[g*32 + j/4] >> ((j & 3)*2)) & 3;
            dst[g*128 + j] = (float)(q8 - 1)*s;
        }
    }
}

// RoPE with YaRN (transliterated from the golden model / llama.cpp).
// Soft-float transcendentals dominate the CPU cost, so everything position-independent is
// precomputed once (rope_init), and cos/sin are computed once per position (rope_pos) and
// shared by all heads and by the Q and K applications.
static float *rope_freq_i;  // Per pair: interpolated angular frequency.
static float *rope_freq_e;  // Per pair: extrapolated (ramp-blended in).
static float  rope_mscale;
static float *rope_c, *rope_s; // Per-pair cos/sin for the current position.
static int    rope_pos_cached;

static float yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base)
{
    return n_dims*logf(n_ctx_orig/(n_rot*2.0f*(float)M_PI))/(2.0f*logf(base));
}

void rope_init(void)
{
    // Bonsai GGUFs use yarn factor 4; tiny sim models use plain rope (vocab heuristic until
    // the BONS header carries yarn fields).
    float theta_base = (float)H->rope_theta;
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    if (H->vocab > 100000) {
        freq_scale = 0.25f;
        ext_factor = 1.0f;
    }
    int   D   = head_dim;
    float low = 0, high = D - 1;
    rope_mscale = 1.0f;
    if (ext_factor != 0.0f) {
        int orig = (n_embd >= 4096) ? 16384 : 8192;
        low  = floorf(yarn_corr_dim(D, orig, 32.0f, theta_base));
        high = ceilf(yarn_corr_dim(D, orig, 1.0f, theta_base));
        if (low  < 0)     low  = 0;
        if (high > D - 1) high = D - 1;
        rope_mscale *= 1.0f + 0.1f*logf(1.0f/freq_scale);
    }
    rope_freq_i = halloc((D/2)*sizeof(float));
    rope_freq_e = halloc((D/2)*sizeof(float));
    rope_c      = halloc((D/2)*sizeof(float));
    rope_s      = halloc((D/2)*sizeof(float));
    for (int p2 = 0; p2 < D/2; p2++) {
        float fe   = powf(theta_base, -2.0f*p2/D); // Extrapolated frequency.
        float ramp = 0.0f;
        if (ext_factor != 0.0f) {
            float y = (p2 - low)/((high - low) > 0.001f ? (high - low) : 0.001f);
            ramp    = (1.0f - fminf(1.0f, fmaxf(0.0f, y)))*ext_factor;
        }
        // theta(pos) = pos * (freq_scale*fe*(1-ramp) + fe*ramp).
        rope_freq_i[p2] = freq_scale*fe*(1.0f - ramp) + fe*ramp;
        rope_freq_e[p2] = fe;
    }
    rope_pos_cached = -1;
}

static void rope_pos(int pos)
{
    if (pos == rope_pos_cached)
        return;
    for (int p2 = 0; p2 < head_dim/2; p2++) {
        float theta = pos*rope_freq_i[p2];
        rope_c[p2] = cosf(theta)*rope_mscale;
        rope_s[p2] = sinf(theta)*rope_mscale;
    }
    rope_pos_cached = pos;
}

COLD void rope_apply(float *vec, int n_heads_v, int pos)
{
    int D = head_dim;
    rope_pos(pos);
    for (int hh = 0; hh < n_heads_v; hh++) {
        float *vh = vec + hh*D;
        for (int p2 = 0; p2 < D/2; p2++) {
            float c = rope_c[p2], s = rope_s[p2];
            float x0 = vh[p2], x1 = vh[p2 + D/2];
            vh[p2]       = x0*c - x1*s;
            vh[p2 + D/2] = x0*s + x1*c;
        }
    }
}
