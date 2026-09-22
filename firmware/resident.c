// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// See resident.h. Two lists per projection group:
//
//   prepare:  FETCH src -> [4] per-block Q8 + e8 -> [3] raw int32 rotation + absmax
//             -> [5] whole-vector Q8 + e8            (or [4] alone when not rotated)
//   matvec:   [1] engine matvec per K-segment -> [2] int32 segment sum -> [8] FP32 restore
//             -> STORE dst
//
// The scratchpad map below is the 27B hybrid's own. Regions the resident lists only use
// transiently may overlap the legacy 27B regions (SP_X, SP_Y..SP_H, SP_FET), which other
// hybrid ops (LM head, embeddings) keep using between lists; what must survive between a
// prepare and its matvecs (the Q8 vector, its exponent) and across the forward (the sign
// bits) sits where no legacy hybrid op writes: [0x32000, 0x3D000).
#include <stdio.h>
#include <string.h>

#include <generated/csr.h>
#include <generated/soc.h>
#include <generated/mem.h>
#include <system.h>

#include "resident.h"
#include "engine.h"
#include "model.h"
#include "perf.h"
#include "seq.h"
#include "bonsai_sequence.h"
#include "progs.h"
#include "static_memory.h"

#if defined(BONSAI_HAS_SEQ) && defined(BONSAI_HAS_HADAMARD) && \
    defined(BONSAI_HAS_ACTIVATION_QUANTIZE) && defined(BONSAI_HAS_PROJECTION_RESTORE) && \
    defined(BONSAI_HAS_KVSTORE) && defined(HAS_INVD_CPU_DCACHE_RANGE)
#define RESIDENT_BUILT 1
#else
#define RESIDENT_BUILT 0
#endif

#define RS_Q8B   0x00000u  // Per-block Q8, [4] out / [3] in (transient, <= 17,408 B).
#define RS_Y     0x04800u  // int32 engine results, 96 KB: one chunk of <= 24,576 rows.
#define RS_F     0x1C800u  // FP32 src, then raw int32 rotation, then FP32 result (68 KB).
#define RS_XW    0x32000u  // Whole-vector Q8 (persists from prepare to its matvecs).
#define RS_E8    0x36400u  // Its exponent beat (int32 in lane 0).
#define RS_EXP   0x36440u  // Per-block e8 (int32 per block, <= 32 blocks).
#define RS_MAX   0x364C0u  // Per-block raw absmax.
#define RS_SIGN  0x3C000u  // Sign bits, 128 B per Hadamard block, all widths (persists).
#define RS_SIGN_END 0x3D000u   // = SP_DESC.
#define RS_SP_END   0x32000u   // RS_F's region ends where the persistent Q8 vector starts.

#define RS_ROWS_MAX  24576u    // int32 results per chunk that fit RS_Y.
#define RS_WIDTH_MAX 17408u    // FP32 source that fits RS_F (and Q8 that fits RS_Q8B/RS_XW).
#define RS_DESC0     0         // Descriptor slots 0..15: the hybrid path builds no other list.

static uint8_t initialized;   // The window struct is not BSS: zero it once, here.

struct resident_state *resident_state(void)
{
    struct resident_state *R = (struct resident_state *)BONSAI_RESIDENT_STATE_ADDR;
    if (!initialized) {
        memset(R, 0, sizeof *R);
        R->enabled = 1;
        initialized = 1;
    }
    return R;
}

void resident_stats_reset(void)
{
    struct resident_state *R = resident_state();
    R->prepare_cycles = R->matvec_cycles = 0;
    R->prepares = R->matvecs = R->fallbacks = 0;
}

void resident_enable(int on) { resident_state()->enabled = on ? 1 : 0; }

int resident_available(void)
{
#if RESIDENT_BUILT
    // ABI majors 1; the requantizer (unit 5) needs activation_quantize minor >= 1.
    return (seq_hadamard_abi_read() >> 16) == 1 &&
           (seq_activation_quantize_abi_read() >> 16) == 1 &&
           (seq_activation_quantize_abi_read() & 0xffff) >= 1 &&
           (seq_projection_restore_abi_read() >> 16) == 1;
#else
    return 0;
#endif
}

int resident_active(void)
{
    struct resident_state *R = resident_state();
    return R->ready && R->enabled && !R->disabled;
}

#if RESIDENT_BUILT

static inline uint32_t eng_addr(const void *p)
{
    return (uint32_t)((uintptr_t)p - MAIN_RAM_BASE + BONSAI_MEM_CPU_OFFSET);
}

static void desc_words(uint32_t idx, const struct bns_seq_descriptor *d)
{
    volatile uint32_t *w = (volatile uint32_t *)(SCRATCHPAD_BASE + SP_DESC + idx*64);
    for (int i = 0; i < 16; i++) w[i] = d->word[i];
}

static void desc_fetch(uint32_t idx, uint32_t addr, uint32_t beats, uint32_t sp_dst)
{
    struct engine_op op = {0};
    op.kind = ENGINE_KIND_FETCH; op.wbase = addr; op.wlen = beats; op.dst = sp_dst;
    seq_desc_engine(idx, &op);
}

static void desc_store(uint32_t idx, uint32_t addr, uint32_t beats, uint32_t sp_src)
{
    struct engine_op op = {0};
    op.kind = ENGINE_KIND_STORE; op.wbase = addr; op.wlen = beats; op.xsrc = sp_src;
    seq_desc_engine(idx, &op);
}

// Run a list and report the resident units' failures at once instead of after the
// sequencer's stall timeout: a failed unit parks the list busy until the engine reset.
static int run_list(uint32_t count, const char *what)
{
    struct resident_state *R = resident_state();
    if (seq_wedged) {
        R->disabled = 1;
        return -1;
    }
    seq_list_start(RS_DESC0, count);
    uint32_t spins = 0;
    for (;;) {
        uint32_t st = seq_status_read();
        if (!(st & 1))
            break;
        uint32_t eq = seq_activation_quantize_error_read(), eh = seq_hadamard_error_read(),
                 er = seq_projection_restore_error_read();
        if ((eq | eh | er) || ++spins > 4000000u) {
            printf("resident: %s list failed at dcount %lu (quantize %08lx hadamard %08lx "
                   "restore %08lx eng %02lx)%s - engine soft reset, resident path off\n",
                   what, (unsigned long)(st >> 8), (unsigned long)eq, (unsigned long)eh,
                   (unsigned long)er, (unsigned long)engine_csr_status_read(),
                   (eq | eh | er) ? "" : " [timeout]");
            R->disabled = 1;
            engine_recover();
            return -1;
        }
    }
    return 0;
}

static int width_index(const struct bons_header *H, uint32_t width)
{
    const struct resident_state *R = resident_state();
    for (int i = 0; i < 4; i++)
        if (H->hadamard_widths[i] == width && R->sign_base[i])
            return i;
    return -1;
}

int resident_init(void)
{
    struct resident_state *R = resident_state();
    const struct bons_header *H = model_header();
    R->n_ch = (uint8_t)H->n_channels;
    R->ready = R->disabled = 0;
    memset(R->sign_base, 0, sizeof R->sign_base);
    if (!resident_available())
        return -1;
    int B = H->hadamard_block;
    if (B != (int)BONSAI_HADAMARD_BLOCK) {
        printf("resident: image block %d, hardware %d - off\n", B, (int)BONSAI_HADAMARD_BLOCK);
        return -1;
    }
    // The dense rotation tensors of the image are H*diag(sign) (+-1, scale 1/32): row 0 of
    // the Sylvester matrix is all +1, so its codes are sign+1: 2 for +1, 0 for negate. Read
    // that row (channel 0's first row of every block tensor) once, keep the packed sign
    // bits resident, and the integer FWHT of unit 3 reproduces the dense rotation exactly.
    uint32_t off = RS_SIGN;
    for (int wi = 0; wi < 4; wi++) {
        uint32_t width = H->hadamard_widths[wi];
        if (!width || width % (uint32_t)B)
            continue;
        if (off + (width/(uint32_t)B)*128u > RS_SIGN_END) {
            printf("resident: no room for the %lu-wide sign bits - off\n", (unsigned long)width);
            return -1;
        }
        R->sign_base[wi] = off;
        for (uint32_t b = 0; b < width/(uint32_t)B; b++) {
            const struct bons_entry *e = model_entry(ROLE_HADAMARD, HADAMARD_LAYER(wi, (int)b));
            if (!e || e->kind != KIND_TERNARY || e->cols != (uint32_t)B) {
                printf("resident: rotation tensor %d/%lu missing - off\n", wi, (unsigned long)b);
                return -1;
            }
            uint32_t rb = bons_row_beats(e->cols, e->kind, e->group);
            uint32_t sb = bons_scale_beats(e->cols, e->group);
            engine_fetch(0, BONSAI_CHAN_WEIGHT_BASE + e->chan_base_beats*32, rb, RS_Y);
            const volatile uint32_t *codes = (const volatile uint32_t *)(SCRATCHPAD_BASE + RS_Y + sb*32);
            volatile uint32_t *signs = (volatile uint32_t *)(SCRATCHPAD_BASE + off);
            for (int j = 0; j < B; j += 32) {
                // 32 codes = two words of sixteen 2-bit codes; code 0 (weight -1) negates.
                uint32_t word = 0, lo = codes[j/16], hi = codes[j/16 + 1];
                for (int k = 0; k < 16; k++) {
                    if (((lo >> (2*k)) & 3u) == 0) word |= 1u << k;
                    if (((hi >> (2*k)) & 3u) == 0) word |= 1u << (16 + k);
                }
                signs[j/32] = word;
            }
            off += 128;
        }
    }
    R->ready = 1;
    return 0;
}

int resident_prepare(const float *src, uint32_t width, int rotate)
{
    struct resident_state *R = resident_state();
    const struct bons_header *H = model_header();
    if (!resident_active() || width > RS_WIDTH_MAX || width % 32 || ((uintptr_t)src & 31)) {
        R->fallbacks++;
        return -1;
    }
    int wi = -1;
    uint32_t B = (uint32_t)H->hadamard_block;
    if (rotate) {
        wi = width_index(H, width);
        if (wi < 0 || width/B > 32) {
            R->fallbacks++;
            return -1;
        }
    }
    uint64_t t0 = perf_now64();
    // The CPU wrote src through its write-back L1: make it visible to the engine's FETCH.
#ifdef HAS_CLEAN_CPU_DCACHE_RANGE
    clean_cpu_dcache_range((void *)src, width*4);
#endif
    struct bns_seq_descriptor d;
    uint32_t n = 0;
    desc_fetch(n++, eng_addr(src), width*4/32, RS_F);
    if (rotate) {
        // unit 4's `elements` is PER BLOCK (gateware: total = elements*blocks), so the
        // rotated form asks for width/B independent 1,024-element exponents, one per
        // Hadamard block. Passing the whole width here quantizes width*blocks elements
        // with the wrong block boundaries and reads past the vector.
        uint32_t blocks = width/B;
        bns_seq_desc_quantize(&d, RS_F, RS_Q8B, RS_EXP, B, blocks);          desc_words(n++, &d);
        bns_seq_desc_hadamard(&d, RS_Q8B, R->sign_base[wi], RS_F, RS_MAX, blocks, 0); desc_words(n++, &d);
        bns_seq_desc_requantize(&d, RS_F, RS_XW, RS_EXP, RS_MAX, RS_E8, blocks); desc_words(n++, &d);
    } else {
        bns_seq_desc_quantize(&d, RS_F, RS_XW, RS_E8, width, 1);              desc_words(n++, &d);
    }
    seq_desc_stop(n++);
    int rc = run_list(n, "prepare");
    R->prepare_cycles += perf_now64() - t0;
    R->prepares++;
    return rc;
}

int resident_matvec(const struct bons_entry *e, float *dst)
{
    struct resident_state *R = resident_state();
    if (!resident_active() || !e || ((uintptr_t)dst & 31) || e->rows % 8 ||
        e->rows > RS_ROWS_MAX || e->rows % (uint32_t)R->n_ch ||
        (e->kind != KIND_TERNARY && e->kind != KIND_Q8)) {
        R->fallbacks++;
        return -1;
    }
    int nseg = (e->flags & FLAG_KSEG) ? model_entry_nkseg(e) : 1;
    if (nseg < 1 || nseg > 4) {
        R->fallbacks++;
        return -1;
    }
    uint32_t rows = e->rows;
    uint32_t ystride = ((rows*4 + 31) & ~31u);   // One int32 result stream per segment.
    if ((uint64_t)nseg*ystride > RS_F - RS_Y || rows*4 > RS_SP_END - RS_F) {
        R->fallbacks++;                          // The segment partials or the FP32 result
        return -1;                               // would not fit their resident regions.
    }
    uint64_t t0 = perf_now64();
    uint32_t rpc = rows/(uint32_t)R->n_ch;
    uint32_t n = 0, xoff = 0;
    for (int k = 0; k < nseg; k++) {
        const struct bons_entry *seg = nseg > 1 ? model_entry_kseg(e, k) : e;
        uint32_t rb = bons_row_beats(seg->cols, seg->kind, seg->group);
        struct engine_op op = {0};
        op.kind   = (seg->kind == KIND_Q8) ? ENGINE_KIND_Q8 : ENGINE_KIND_TERNARY;
        op.rows   = rpc;
        op.cols   = seg->cols;
        op.wbase  = BONSAI_CHAN_WEIGHT_BASE + seg->chan_base_beats*32;
        op.wlen   = rpc*rb;
        op.sbeats = bons_scale_beats(seg->cols, seg->group);
        op.cbeats = bons_code_beats(seg->cols, seg->kind);
        op.bias   = seg->bias;
        op.rshift = seg->rshift;
        op.xsrc   = RS_XW + xoff;
        op.dst    = RS_Y + (uint32_t)k*ystride;
        seq_desc_engine(n++, &op);
        if (k) {
            // Wrapping int32 sum of the segment partials, as the fused path's reduce_acc.
            int32_t a[4] = {(int32_t)RS_Y, (int32_t)(RS_Y + (uint32_t)k*ystride),
                            (int32_t)RS_Y, (int32_t)((rows + 7)/8)};
            seq_desc_vec(n++, PROG_REDUCE_ACC, a, 4, 0);
        }
        xoff += seg->cols;
    }
    struct bns_seq_descriptor d;
    if (bns_seq_desc_restore(&d, RS_Y, RS_F, RS_E8, rows, (int32_t)e->rshift - (int32_t)e->bias)) {
        R->fallbacks++;
        return -1;
    }
    desc_words(n++, &d);
    desc_store(n++, eng_addr(dst), rows*4/32, RS_F);
    seq_desc_stop(n++);
    int rc = run_list(n, "matvec");
    if (!rc)
        invd_cpu_dcache_range(dst, rows*4);
    R->matvec_cycles += perf_now64() - t0;
    R->matvecs++;
    return rc;
}

#else  // !RESIDENT_BUILT

int resident_init(void) { resident_state()->ready = 0; return -1; }
int resident_prepare(const float *src, uint32_t width, int rotate)
{
    (void)src; (void)width; (void)rotate; resident_state()->fallbacks++; return -1;
}
int resident_matvec(const struct bons_entry *e, float *dst)
{
    (void)e; (void)dst; resident_state()->fallbacks++; return -1;
}

#endif
