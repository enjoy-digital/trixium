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
#include "fastmath.h"

#if defined(BONSAI_HAS_SEQ) && defined(BONSAI_HAS_HADAMARD) && \
    defined(BONSAI_HAS_ACTIVATION_QUANTIZE) && defined(BONSAI_HAS_PROJECTION_RESTORE) && \
    defined(BONSAI_HAS_KVSTORE) && defined(HAS_INVD_CPU_DCACHE_RANGE) && \
    defined(HAS_FLUSH_CPU_DCACHE_RANGE) && defined(HAS_CLEAN_CPU_DCACHE_RANGE)
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
#define RS_DESC_MAX  64u      // of the 192 slots in SP_DESC.

// Unit 11 (SiLU-multiply). The FFN activation runs gate -> RS_F, up -> RS_Y (restored in
// place), SiLU in place over RS_F; its fixup bitmap lands in RS_Q8B, which only a prepare
// uses. The 1,025-beat table image is staged in RS_F once, at init.
#if defined(BONSAI_HAS_SILU)
#define SILU_BUILT RESIDENT_BUILT
#else
#define SILU_BUILT 0
#endif
#define RS_FIXMAP RS_Q8B

void resident_listprof_reset(void);
static uint8_t initialized;   // The window struct is not BSS: zero it once, here.

_Static_assert(sizeof(struct resident_state) <= 256, "resident_state window is 256 bytes");

struct resident_state *resident_state(void)
{
    struct resident_state *R = (struct resident_state *)BONSAI_RESIDENT_STATE_ADDR;
    if (!initialized) {
        memset(R, 0, sizeof *R);
        R->enabled = 1;
        R->silu_enabled = 1;
        initialized = 1;
    }
    return R;
}

void resident_stats_reset(void)
{
    struct resident_state *R = resident_state();
    R->prepare_cycles = R->matvec_cycles = 0;
    R->prepares = R->matvecs = R->fallbacks = 0;
    R->silu_cycles = 0;
    resident_listprof_reset();
    R->silu_lists = R->silu_fixups = 0;
}

void resident_enable(int on) { resident_state()->enabled = on ? 1 : 0; }
void resident_silu_enable(int on) { resident_state()->silu_enabled = on ? 1 : 0; }
void resident_mark_act(void) { resident_state()->act_resident = 1; }

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
// Per-descriptor list profile (console `stats`): cycles from one completed descriptor to
// the next, accumulated per list name (the `what` pointer) and descriptor index. The
// completion poll below reads the sequencer anyway; one timer read per poll is the cost.
struct list_prof {
    const char *what;
    uint32_t calls, n;
    uint64_t cyc[RS_DESC_MAX];
};
#define LISTPROF_N 16
#define LP ((struct list_prof *)BONSAI_LISTPROF_ADDR)

void resident_listprof_reset(void)
{
    memset(LP, 0, sizeof(struct list_prof)*LISTPROF_N);
}

void resident_listprof_print(uint32_t forwards)
{
    for (int i = 0; i < LISTPROF_N && LP[i].what; i++) {
        struct list_prof *p = &LP[i];
        if (!p->calls || !forwards) continue;
        uint64_t total = 0;
        for (uint32_t d = 0; d < p->n; d++) total += p->cyc[d];
        printf("listprof %s: %lu calls/fwd, %lu cycles/call:", p->what,
               (unsigned long)(p->calls/forwards), (unsigned long)(total/p->calls));
        for (uint32_t d = 0; d < p->n; d++)
            printf(" %lu", (unsigned long)(p->cyc[d]/p->calls));
        puts("");
    }
}

static struct list_prof *listprof_slot(const char *what)
{
    for (int i = 0; i < LISTPROF_N; i++) {
        if (LP[i].what == what) return &LP[i];
        if (!LP[i].what) { LP[i].what = what; return &LP[i]; }
    }
    return 0;
}

static int run_list(uint32_t count, const char *what)
{
    struct resident_state *R = resident_state();
    struct list_prof *prof = listprof_slot(what);
    uint32_t last_d = 0;
    uint64_t last_t = perf_now64();
    if (seq_wedged) {
        R->disabled = 1;
        return -1;
    }
    R->act_resident = 0;   // Every list may rewrite RS_F; resident_ffn_act re-arms it.
    seq_list_start(RS_DESC0, count);
    uint32_t spins = 0;
    for (;;) {
        uint32_t st = seq_status_read();
        if (prof) {
            uint32_t dc = st >> 8;
            if (dc != last_d && dc < RS_DESC_MAX) {
                uint64_t now = perf_now64();
                prof->cyc[last_d] += now - last_t;       // descriptor last_d (and any skipped)
                last_t = now; last_d = dc;
            }
        }
        if (!(st & 1)) {
            if (prof) {
                if (last_d < RS_DESC_MAX) prof->cyc[last_d] += perf_now64() - last_t;
                prof->calls++;
                if (count > prof->n) prof->n = count;
            }
            break;
        }
        uint32_t eq = seq_activation_quantize_error_read(), eh = seq_hadamard_error_read(),
                 er = seq_projection_restore_error_read(), es = 0;
#if SILU_BUILT
        es = seq_silu_error_read();
#endif
#ifdef CSR_SEQ_NORM_ERROR_ADDR
        es |= seq_norm_error_read();
#endif
#ifdef CSR_SEQ_CONV_ERROR_ADDR
        es |= seq_conv_error_read();
#endif
        if ((eq | eh | er | es) || ++spins > 4000000u) {
            printf("resident: %s list failed at dcount %lu (quantize %08lx hadamard %08lx "
                   "restore %08lx silu/norm/conv %08lx eng %02lx)%s - engine soft reset, resident path off\n",
                   what, (unsigned long)(st >> 8), (unsigned long)eq, (unsigned long)eh,
                   (unsigned long)er, (unsigned long)es, (unsigned long)engine_csr_status_read(),
                   (eq | eh | er | es) ? "" : " [timeout]");
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
    if (!resident_active() || width > RS_WIDTH_MAX || width % 32 || ((uintptr_t)src & 31) ||
        (!src && !R->act_resident)) {
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
    struct bns_seq_descriptor d;
    uint32_t n = 0;
    if (src) {
        // The CPU wrote src through its write-back L1: make it visible to the engine's
        // FETCH. Clean, not flush: a clean line cannot clobber what the engine reads back.
        clean_cpu_dcache_range((void *)src, width*4);
        desc_fetch(n++, eng_addr(src), width*4/32, RS_F);
    }
    // src == NULL: the FP32 source is already in RS_F (resident_ffn_act) - consumed here.
    R->act_resident = 0;
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

// Descriptors for dst_sp = FP32(e * resident Q8): engine matvec per K-segment, int32
// segment sum, restore into dst_sp (RS_F, or RS_Y in place). Appends at *n; -1 = the
// tensor or its results do not fit the resident map (nothing is run).
static int build_matvec_rows(uint32_t *n, const struct bons_entry *e, uint32_t row0,
                             uint32_t rows, uint32_t ystage, uint32_t dst_sp);
static int build_matvec_at(uint32_t *n, const struct bons_entry *e, uint32_t ystage, uint32_t dst_sp)
{
    return e ? build_matvec_rows(n, e, 0, e->rows, ystage, dst_sp) : -1;
}
static int build_matvec(uint32_t *n, const struct bons_entry *e, uint32_t dst_sp)
{
    return build_matvec_at(n, e, RS_Y, dst_sp);
}

// Rows [row0, row0 + rows) of e (row0 and rows multiples of the channel count: channel c
// holds rows c, c + n_ch, ...). ystage: where the int32 segment partials go (RS_Y, or
// another region the caller keeps clear of RS_F and the persistent map); dst_sp: RS_F, or
// ystage itself (restore in place).
static int build_matvec_rows(uint32_t *n, const struct bons_entry *e, uint32_t row0,
                             uint32_t rows, uint32_t ystage, uint32_t dst_sp)
{
    struct resident_state *R = resident_state();
    if (!e || rows % 8 || rows > RS_ROWS_MAX || rows % (uint32_t)R->n_ch || row0 % (uint32_t)R->n_ch ||
        row0 + rows > e->rows || (e->kind != KIND_TERNARY && e->kind != KIND_Q8))
        return -1;
    int nseg = (e->flags & FLAG_KSEG) ? model_entry_nkseg(e) : 1;
    if (nseg < 1 || nseg > 4)
        return -1;
    uint32_t ystride = ((rows*4 + 31) & ~31u);   // One int32 result stream per segment.
    if (ystage % 32 || ystage + (uint64_t)nseg*ystride > RS_F)   // The partials fit below RS_F,
        return -1;
    if (dst_sp == RS_F ? rows*4 > RS_SP_END - RS_F : dst_sp != ystage)   // the result its region,
        return -1;
    if (*n + 2u*(uint32_t)nseg > RS_DESC_MAX - 1u)                       // and the list its slots.
        return -1;
    uint32_t rpc = rows/(uint32_t)R->n_ch;
    uint32_t xoff = 0;
    for (int k = 0; k < nseg; k++) {
        const struct bons_entry *seg = nseg > 1 ? model_entry_kseg(e, k) : e;
        uint32_t rb = bons_row_beats(seg->cols, seg->kind, seg->group);
        struct engine_op op = {0};
        op.kind   = (seg->kind == KIND_Q8) ? ENGINE_KIND_Q8 : ENGINE_KIND_TERNARY;
        op.rows   = rpc;
        op.cols   = seg->cols;
        op.wbase  = BONSAI_CHAN_WEIGHT_BASE + (seg->chan_base_beats + (row0/(uint32_t)R->n_ch)*rb)*32;
        op.wlen   = rpc*rb;
        op.sbeats = bons_scale_beats(seg->cols, seg->group);
        op.cbeats = bons_code_beats(seg->cols, seg->kind);
        op.bias   = seg->bias;
        op.rshift = seg->rshift;
        op.xsrc   = RS_XW + xoff;
        op.dst    = ystage + (uint32_t)k*ystride;
        seq_desc_engine((*n)++, &op);
        if (k) {
            // Wrapping int32 sum of the segment partials, as the fused path's reduce_acc.
            int32_t a[4] = {(int32_t)ystage, (int32_t)(ystage + (uint32_t)k*ystride),
                            (int32_t)ystage, (int32_t)((rows + 7)/8)};
            seq_desc_vec((*n)++, PROG_REDUCE_ACC, a, 4, 0);
        }
        xoff += seg->cols;
    }
    struct bns_seq_descriptor d;
    if (bns_seq_desc_restore(&d, ystage, dst_sp, RS_E8, rows, (int32_t)e->rshift - (int32_t)e->bias))
        return -1;
    desc_words((*n)++, &d);
    return 0;
}

// The destination is an ordinary hybrid buffer the CPU wrote on earlier forwards, so it
// can hold DIRTY lines. One evicted while the engine's STORE is in flight would overwrite
// the result in DDR, and the invalidate after the list would then expose the stale value -
// rare, data-dependent corruption. Clean AND invalidate before the store, so the CPU holds
// no copy of this span at all. (infer.c's i32_stage needs only the invalidate: nothing but
// the engine ever writes it.)
static void store_out(uint32_t *n, float *dst, uint32_t rows, uint32_t sp_src)
{
    flush_cpu_dcache_range(dst, rows*4);
    desc_store((*n)++, eng_addr(dst), (rows*4 + 31)/32, sp_src);
}

int resident_matvec(const struct bons_entry *e, float *dst)
{
    struct resident_state *R = resident_state();
    uint32_t n = 0;
    if (!resident_active() || ((uintptr_t)dst & 31) || build_matvec(&n, e, RS_F)) {
        R->fallbacks++;
        return -1;
    }
    uint64_t t0 = perf_now64();
    store_out(&n, dst, e->rows, RS_F);
    seq_desc_stop(n++);
    int rc = run_list(n, "matvec");
    if (!rc)
        invd_cpu_dcache_range(dst, e->rows*4);
    R->matvec_cycles += perf_now64() - t0;
    R->matvecs++;
    return rc;
}

// General lists (resident.h, "list builder"): hybrid.c composes whole-layer lists from
// these. Every main-RAM transfer does its own cache maintenance; stores are invalidated
// after a successful run.
static struct {
    uint32_t n, ninv, bad;
    void *inv[16];
    uint32_t inv_bytes[16];
} RL;

void rl_begin(void) { RL.n = RL.ninv = RL.bad = 0; }

static int rl_room(uint32_t k)
{
    if (RL.n + k + 1 > RS_DESC_MAX) RL.bad = 1;
    return !RL.bad;
}

void rl_fetch(const void *src, uint32_t bytes, uint32_t sp)
{
    if (!rl_room(1) || ((uintptr_t)src & 31) || bytes % 32 || sp % 32) { RL.bad = 1; return; }
    clean_cpu_dcache_range((void *)src, bytes);
    desc_fetch(RL.n++, eng_addr(src), bytes/32, sp);
}

void rl_store(void *dst, uint32_t bytes, uint32_t sp)
{
    if (!rl_room(1) || ((uintptr_t)dst & 31) || bytes % 32 || sp % 32 || RL.ninv >= 16) { RL.bad = 1; return; }
    flush_cpu_dcache_range(dst, bytes);
    desc_store(RL.n++, eng_addr(dst), bytes/32, sp);
    RL.inv[RL.ninv] = dst; RL.inv_bytes[RL.ninv++] = bytes;
}

void rl_matvec(const struct bons_entry *e, uint32_t ystage, uint32_t dst_sp)
{
    if (!rl_room(9) || build_matvec_at(&RL.n, e, ystage, dst_sp)) RL.bad = 1;
}

void rl_matvec_rows(const struct bons_entry *e, uint32_t row0, uint32_t rows, uint32_t ystage,
                    uint32_t dst_sp)
{
    if (!rl_room(9) || build_matvec_rows(&RL.n, e, row0, rows, ystage, dst_sp)) RL.bad = 1;
}

void rl_desc(const struct bns_seq_descriptor *d)
{
    if (!rl_room(1)) return;
    desc_words(RL.n++, d);
}

void rl_engine(const struct engine_op *op)
{
    if (!rl_room(1)) return;
    seq_desc_engine(RL.n++, op);
}

void rl_prepare(const float *src, uint32_t width, int rotate)
{
    struct resident_state *R = resident_state();
    const struct bons_header *H = model_header();
    uint32_t B = (uint32_t)H->hadamard_block;
    struct bns_seq_descriptor d;
    if (!rl_room(4) || width > RS_WIDTH_MAX || width % 32) { RL.bad = 1; return; }
    if (src)
        rl_fetch(src, width*4, RS_F);
    if (rotate) {
        int wi = width_index(H, width);
        if (wi < 0 || width/B > 32) { RL.bad = 1; return; }
        uint32_t blocks = width/B;
        bns_seq_desc_quantize(&d, RS_F, RS_Q8B, RS_EXP, B, blocks);          desc_words(RL.n++, &d);
        bns_seq_desc_hadamard(&d, RS_Q8B, R->sign_base[wi], RS_F, RS_MAX, blocks, 0); desc_words(RL.n++, &d);
        bns_seq_desc_requantize(&d, RS_F, RS_XW, RS_EXP, RS_MAX, RS_E8, blocks); desc_words(RL.n++, &d);
    } else {
        bns_seq_desc_quantize(&d, RS_F, RS_XW, RS_E8, width, 1);              desc_words(RL.n++, &d);
    }
}

int rl_run(const char *what)
{
    if (RL.bad || !resident_active())
        return -1;
    seq_desc_stop(RL.n++);
    int rc = run_list(RL.n, what);
    if (!rc)
        for (uint32_t i = 0; i < RL.ninv; i++)
            invd_cpu_dcache_range(RL.inv[i], RL.inv_bytes[i]);
    return rc;
}

#if SILU_BUILT

// Recompute the lanes the unit flagged (they hold g): the firmware's own expression, into
// RS_F and, when given, the main-RAM copy.
static uint32_t silu_patch(uint32_t rows, float *act)
{
    struct resident_state *R = resident_state();
    const volatile uint32_t *bits = (const volatile uint32_t *)(SCRATCHPAD_BASE + RS_FIXMAP);
    volatile float *f = (volatile float *)(SCRATCHPAD_BASE + RS_F);
    const volatile float *m = (const volatile float *)(SCRATCHPAD_BASE + RS_Y);
    uint32_t patched = 0;
    for (uint32_t w = 0; w < (rows + 31)/32; w++) {
        uint32_t word = bits[w];
        while (word) {
            uint32_t i = 32*w + (uint32_t)__builtin_ctz(word);
            word &= word - 1;
            float r = m[i]*bonsai_silu_lookup(f[i], R->silu_lut);
            f[i] = r;
            if (act) act[i] = r;
            patched++;
        }
    }
    return patched;
}

// The general form for list builders (resident.h): g at g_sp, m at m_sp (0: m = 1.0), the
// result at dst_sp with the unit's optional segment permutation, the bitmap at fixmap_sp.
uint32_t resident_silu_patch(uint32_t g_sp, uint32_t m_sp, uint32_t dst_sp, uint32_t fixmap_sp,
                             uint32_t n, uint32_t seg, uint32_t perm_a, uint32_t perm_b)
{
    struct resident_state *R = resident_state();
    const volatile uint32_t *bits = (const volatile uint32_t *)(SCRATCHPAD_BASE + fixmap_sp);
    const volatile float *g = (const volatile float *)(SCRATCHPAD_BASE + g_sp);
    const volatile float *m = (const volatile float *)(SCRATCHPAD_BASE + m_sp);
    volatile float *dst = (volatile float *)(SCRATCHPAD_BASE + dst_sp);
    uint32_t patched = 0;
    for (uint32_t w = 0; w < (n + 31)/32; w++) {
        uint32_t word = bits[w];
        while (word) {
            uint32_t i = 32*w + (uint32_t)__builtin_ctz(word);
            word &= word - 1;
            uint32_t o = i;
            if (perm_a > 1) {
                uint32_t sg = i/seg, within = i - sg*seg;
                o = ((sg % perm_a)*perm_b + sg/perm_a)*seg + within;
            }
            float mv = m_sp ? m[i] : 1.0f;
            dst[o] = mv*bonsai_silu_lookup(g[i], R->silu_lut);
            patched++;
        }
    }
    return patched;
}

// End-to-end check at boot: gates across every range (table, both polynomial tails, 16,
// the pass-through and -0 ranges, specials) times ordinary, huge and subnormal
// multipliers, through the unit and the fixup patch, against the CPU element by element.
#define SILU_SELFTEST_N 640u
static int silu_selftest(void)
{
    struct resident_state *R = resident_state();
    volatile uint32_t *g = (volatile uint32_t *)(SCRATCHPAD_BASE + RS_F);
    volatile uint32_t *m = (volatile uint32_t *)(SCRATCHPAD_BASE + RS_Y);
    union { float f; uint32_t u; } a, b, c;
    static const uint32_t special[8] = {0x7f800000u, 0xff800000u, 0x7fc00000u, 0x00000001u,
                                        0x80000000u, 0x41800000u, 0xc1800000u, 0xc2a00000u};
    static const uint32_t mult[8] = {0x3f800000u, 0xbfa00000u, 0x40490fdbu, 0x3a83126fu,
                                     0x7e967699u, 0x00000003u, 0x0da24260u, 0x80000000u};
    for (uint32_t i = 0; i < SILU_SELFTEST_N; i++) {
        a.f = -100.0f + (float)i*0.3125f;             // -100 .. 99.7: every range
        g[i] = (i % 61 == 7) ? special[(i/61) % 8] : a.u;
        m[i] = mult[i % 8];
    }
    uint32_t lost = 0;
    for (uint32_t i = 0; i < SILU_SELFTEST_N; i++) {
        a.f = -100.0f + (float)i*0.3125f;
        if (g[i] != ((i % 61 == 7) ? special[(i/61) % 8] : a.u) || m[i] != mult[i % 8])
            lost++;
    }
    if (lost)
        printf("resident: silu self-test inputs: %lu of %lu CPU-written elements read back wrong\n",
               (unsigned long)lost, (unsigned long)SILU_SELFTEST_N);
    struct bns_seq_descriptor d;
    uint32_t n = 0;
    bns_seq_desc_silu(&d, RS_F, RS_Y, RS_F, RS_FIXMAP, SILU_SELFTEST_N);
    desc_words(n++, &d);
    seq_desc_stop(n++);
    // The inputs are regenerated below; RS_F is overwritten in place.
    if (run_list(n, "silu self-test"))
        return -1;
    uint32_t fixups = seq_silu_fixups_read(), patched = silu_patch(SILU_SELFTEST_N, NULL);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < SILU_SELFTEST_N; i++) {
        a.f = -100.0f + (float)i*0.3125f;
        a.u = (i % 61 == 7) ? special[(i/61) % 8] : a.u;
        b.u = mult[i % 8];
        c.f = b.f*bonsai_silu_lookup(a.f, R->silu_lut);
        if (g[i] != c.u && bad++ < 4)
            printf("resident: silu self-test element %lu g=%08lx m=%08lx: unit %08lx, cpu %08lx\n",
                   (unsigned long)i, (unsigned long)a.u, (unsigned long)b.u,
                   (unsigned long)g[i], (unsigned long)c.u);
    }
    printf("resident: silu self-test %lu elements, %lu fixups (%lu patched), %lu mismatches\n",
           (unsigned long)SILU_SELFTEST_N, (unsigned long)fixups, (unsigned long)patched,
           (unsigned long)bad);
    return (bad || fixups != patched || !fixups) ? -1 : 0;
}

#define SILU_IMAGE_WORDS (1025u*8u)

// Diagnostic, run when the self-test fails: every table node g = -16 + k/128 (F = 0, so
// the unit must return lut[k] exactly for m = 1) and every midpoint (F = 1/2, the slopes),
// inputs DMA-staged, results against the CPU. Prints which indices fail and their pattern.
static void silu_probe(uint32_t *stage, int half)
{
    struct resident_state *R = resident_state();
    const uint32_t N = BONSAI_SILU_LUT_N;           // k = 0 (g = -16) is a fixup; skipped.
    union { float f; uint32_t u; } a, c;
    for (uint32_t k = 0; k < N; k++) {
        a.f = -16.0f + ((float)k + (half ? 0.5f : 0.0f))*(1.0f/128.0f);
        stage[k] = a.u;
        stage[N + k] = 0x3f800000u;
    }
    clean_cpu_dcache_range(stage, 2*N*4);
    struct bns_seq_descriptor d;
    uint32_t n = 0;
    desc_fetch(n++, eng_addr(stage), N*4/32, RS_F);
    desc_fetch(n++, eng_addr(stage + N), N*4/32, RS_Y);
    bns_seq_desc_silu(&d, RS_F, RS_Y, RS_F, RS_FIXMAP, N);
    desc_words(n++, &d);
    seq_desc_stop(n++);
    if (run_list(n, "silu probe"))
        return;
    const volatile uint32_t *r = (const volatile uint32_t *)(SCRATCHPAD_BASE + RS_F);
    uint32_t bad = 0, lo = 0, hi = 0, lane[8] = {0}, printed = 0;
    for (uint32_t k = 1; k < N; k++) {
        a.f = -16.0f + ((float)k + (half ? 0.5f : 0.0f))*(1.0f/128.0f);
        c.f = bonsai_silu_lookup(a.f, R->silu_lut);
        if (r[k] == c.u)
            continue;
        if (!bad++) lo = k;
        hi = k;
        lane[k % 8]++;
        if (printed++ < 6)
            printf("  probe%s k=%lu g=%08lx: unit %08lx, cpu %08lx\n", half ? "-mid" : "",
                   (unsigned long)k, (unsigned long)a.u, (unsigned long)r[k], (unsigned long)c.u);
    }
    printf("resident: silu probe %s: %lu of %lu wrong, k %lu..%lu, per lane %lu %lu %lu %lu %lu %lu %lu %lu\n",
           half ? "midpoints" : "nodes", (unsigned long)bad, (unsigned long)(N - 1),
           (unsigned long)lo, (unsigned long)hi,
           (unsigned long)lane[0], (unsigned long)lane[1], (unsigned long)lane[2], (unsigned long)lane[3],
           (unsigned long)lane[4], (unsigned long)lane[5], (unsigned long)lane[6], (unsigned long)lane[7]);
}

int resident_silu_init(const float *lut, uint32_t *stage)
{
    struct resident_state *R = resident_state();
    R->silu_ready = 0;
    R->silu_lut = lut;
    if (!R->ready || !stage || ((uintptr_t)stage & 31) || (seq_silu_abi_read() >> 16) != 1)
        return -1;
    // The table image the unit loads: lut[0..4096] (lut[4096] alone in beat 512) and the
    // interpolation slopes lut[i+1] - lut[i], one FP32 subtraction each - exactly what
    // bonsai_silu_lookup computes per element. Built in main RAM and DMA-fetched into RS_F
    // in the same list as the LOAD, like every other bulk transfer, then read back: a bad
    // image is reported as such instead of as self-test mismatches. (v13's 82 self-test
    // mismatches were the kernel tables exported over the CSR bus - see silu_mul.py.)
    union { float f; uint32_t u; } v;
    for (uint32_t i = 0; i < SILU_IMAGE_WORDS; i++) stage[i] = 0;
    for (int i = 0; i <= BONSAI_SILU_LUT_N; i++) { v.f = lut[i]; stage[i] = v.u; }
    for (int i = 0; i < BONSAI_SILU_LUT_N; i++) { v.f = lut[i + 1] - lut[i]; stage[513*8 + i] = v.u; }
    clean_cpu_dcache_range(stage, SILU_IMAGE_WORDS*4);
    struct bns_seq_descriptor d;
    uint32_t n = 0;
    desc_fetch(n++, eng_addr(stage), SILU_IMAGE_WORDS/8, RS_F);
    bns_seq_desc_silu_load(&d, RS_F);
    desc_words(n++, &d);
    seq_desc_stop(n++);
    if (run_list(n, "silu table load"))
        return -1;
    // What the unit read must be the image: compare the scratchpad copy word for word.
    const volatile uint32_t *w = (const volatile uint32_t *)(SCRATCHPAD_BASE + RS_F);
    uint32_t bad = 0, first = 0;
    for (uint32_t i = 0; i < SILU_IMAGE_WORDS; i++)
        if (w[i] != stage[i] && !bad++) first = i;
    if (bad) {
        printf("resident: silu table image in the scratchpad differs in %lu words (first %lu) - off\n",
               (unsigned long)bad, (unsigned long)first);
        return -1;
    }
    if (silu_selftest()) {
        silu_probe(stage, 0);
        silu_probe(stage, 1);
        return -1;
    }
    R->silu_ready = 1;
    R->silu_image = stage;          // kept: rl_gate restores the SiLU table from it
    return 0;
}

int resident_silu_active(void)
{
    struct resident_state *R = resident_state();
    return resident_active() && R->silu_ready && R->silu_enabled;
}

int resident_gate_init(const float *sig_lut, uint32_t *image)
{
    struct resident_state *R = resident_state();
    R->gate_ready = 0;
    if (!R->silu_ready || !R->silu_image || !sig_lut || !image || ((uintptr_t)image & 31) ||
        (seq_silu_abi_read() >> 16) != 1 || (seq_silu_abi_read() & 0xffff) < 2)
        return -1;
    // The SiLU image's layout (resident_silu_init) on the canonical sigmoid table.
    union { float f; uint32_t u; } v;
    for (uint32_t i = 0; i < SILU_IMAGE_WORDS; i++) image[i] = 0;
    for (int i = 0; i <= BONSAI_SILU_LUT_N; i++) { v.f = sig_lut[i]; image[i] = v.u; }
    for (int i = 0; i < BONSAI_SILU_LUT_N; i++) { v.f = sig_lut[i + 1] - sig_lut[i]; image[513*8 + i] = v.u; }
    clean_cpu_dcache_range(image, SILU_IMAGE_WORDS*4);
    R->sig_image = image;
    R->sig_lut = sig_lut;
    R->gate_ready = 1;
    return 0;
}

int resident_silu_restore(void)
{
    // After a failed gated list the unit may still hold the sigmoid table.
    struct resident_state *R = resident_state();
    struct bns_seq_descriptor d;
    uint32_t n = 0;
    if (!R->silu_ready || !R->silu_image)
        return -1;
    desc_fetch(n++, eng_addr(R->silu_image), SILU_IMAGE_WORDS/8, RS_F);
    bns_seq_desc_silu_load(&d, RS_F);
    desc_words(n++, &d);
    seq_desc_stop(n++);
    if (run_list(n, "silu table restore")) {
        R->silu_enabled = 0;
        puts("resident: the SiLU table could not be restored - unit 11 off");
        return -1;
    }
    return 0;
}

int resident_gate_active(void)
{
    return resident_silu_active() && resident_state()->gate_ready;
}

void rl_gate(uint32_t g_sp, uint32_t m_sp, uint32_t fixmap_sp, uint32_t img_sp, uint32_t n)
{
    struct resident_state *R = resident_state();
    struct bns_seq_descriptor d;
    if (!resident_gate_active() || !rl_room(6)) { RL.bad = 1; return; }
    desc_fetch(RL.n++, eng_addr(R->sig_image), SILU_IMAGE_WORDS/8, img_sp);
    bns_seq_desc_silu_load(&d, img_sp);                          desc_words(RL.n++, &d);
    if (bns_seq_desc_silu_table(&d, g_sp, m_sp, g_sp, fixmap_sp, n)) { RL.bad = 1; return; }
    desc_words(RL.n++, &d);
    desc_fetch(RL.n++, eng_addr(R->silu_image), SILU_IMAGE_WORDS/8, img_sp);
    bns_seq_desc_silu_load(&d, img_sp);                          desc_words(RL.n++, &d);
}

uint32_t resident_gate_patch(uint32_t fixmap_sp, const float *g, uint32_t m_sp, float *dst,
                             uint32_t n)
{
    struct resident_state *R = resident_state();
    const volatile uint32_t *bits = (const volatile uint32_t *)(SCRATCHPAD_BASE + fixmap_sp);
    const volatile float *m = (const volatile float *)(SCRATCHPAD_BASE + m_sp);
    uint32_t patched = 0;
    for (uint32_t w = 0; w < (n + 31)/32; w++) {
        uint32_t word = bits[w];
        while (word) {
            uint32_t i = 32*w + (uint32_t)__builtin_ctz(word);
            word &= word - 1;
            dst[i] = m[i]*bonsai_sigmoid_lookup(g[i], R->sig_lut);
            patched++;
        }
    }
    R->gate_fixups += patched;
    return patched;
}

int resident_ffn_act(const struct bons_entry *gate, const struct bons_entry *up,
                     float *act, float *gate_out, float *up_out)
{
    struct resident_state *R = resident_state();
    uint32_t n = 0;
    if (!resident_silu_active() || !gate || !up || gate->rows != up->rows ||
        gate->rows > RS_WIDTH_MAX || ((uintptr_t)act & 31) ||
        ((uintptr_t)gate_out & 31) || ((uintptr_t)up_out & 31)) {
        R->fallbacks++;
        return -1;
    }
    uint32_t rows = gate->rows;
    uint64_t t0 = perf_now64();
    struct bns_seq_descriptor d;
    if (build_matvec(&n, gate, RS_F) || (gate_out && n > RS_DESC_MAX - 6u)) goto decline;
    if (gate_out) store_out(&n, gate_out, rows, RS_F);
    if (build_matvec(&n, up, RS_Y) || n + (up_out ? 4u : 3u) > RS_DESC_MAX) goto decline;
    if (up_out) store_out(&n, up_out, rows, RS_Y);
    bns_seq_desc_silu(&d, RS_F, RS_Y, RS_F, RS_FIXMAP, rows);
    desc_words(n++, &d);
    // The activation also goes to main RAM: the down projection's CPU fallback and the
    // traces read it there. Fixup lanes are patched in both copies below.
    store_out(&n, act, rows, RS_F);
    seq_desc_stop(n++);
    int rc = run_list(n, "ffn activation");
    if (!rc) {
        invd_cpu_dcache_range(act, rows*4);
        if (gate_out) invd_cpu_dcache_range(gate_out, rows*4);
        if (up_out) invd_cpu_dcache_range(up_out, rows*4);
        // Elements outside the unit's exact range (|gate| tails, specials, non-normal
        // products) kept g in RS_F; recompute them with the firmware's own function.
        uint32_t fixups = seq_silu_fixups_read();
        R->silu_fixups += fixups;
        if (fixups)
            silu_patch(rows, act);
        R->act_resident = 1;
    }
    R->silu_cycles += perf_now64() - t0;
    R->silu_lists++;
    return rc;
decline:
    R->fallbacks++;
    return -1;
}

#else

int resident_silu_init(const float *lut, uint32_t *stage)
{
    (void)stage; resident_state()->silu_lut = lut; return -1;
}
int resident_silu_active(void) { return 0; }
uint32_t resident_silu_patch(uint32_t g_sp, uint32_t m_sp, uint32_t dst_sp, uint32_t fixmap_sp,
                             uint32_t n, uint32_t seg, uint32_t perm_a, uint32_t perm_b)
{
    (void)g_sp; (void)m_sp; (void)dst_sp; (void)fixmap_sp; (void)n; (void)seg; (void)perm_a; (void)perm_b;
    return 0;
}
int resident_ffn_act(const struct bons_entry *gate, const struct bons_entry *up,
                     float *act, float *gate_out, float *up_out)
{
    (void)gate; (void)up; (void)act; (void)gate_out; (void)up_out;
    resident_state()->fallbacks++;
    return -1;
}
int resident_gate_init(const float *sig_lut, uint32_t *image)
{
    (void)sig_lut; (void)image; resident_state()->gate_ready = 0; return -1;
}
int resident_gate_active(void) { return 0; }
int resident_silu_restore(void) { return -1; }
void rl_gate(uint32_t g_sp, uint32_t m_sp, uint32_t fixmap_sp, uint32_t img_sp, uint32_t n)
{
    (void)g_sp; (void)m_sp; (void)fixmap_sp; (void)img_sp; (void)n;
}
uint32_t resident_gate_patch(uint32_t fixmap_sp, const float *g, uint32_t m_sp, float *dst,
                             uint32_t n)
{
    (void)fixmap_sp; (void)g; (void)m_sp; (void)dst; (void)n; return 0;
}

#endif

#else  // !RESIDENT_BUILT

void resident_listprof_reset(void) {}
void resident_listprof_print(uint32_t forwards) { (void)forwards; }

void rl_begin(void) {}
void rl_fetch(const void *src, uint32_t bytes, uint32_t sp) { (void)src; (void)bytes; (void)sp; }
void rl_store(void *dst, uint32_t bytes, uint32_t sp) { (void)dst; (void)bytes; (void)sp; }
void rl_matvec(const struct bons_entry *e, uint32_t ystage, uint32_t dst_sp) { (void)e; (void)ystage; (void)dst_sp; }
void rl_matvec_rows(const struct bons_entry *e, uint32_t row0, uint32_t rows, uint32_t ystage, uint32_t dst_sp)
{ (void)e; (void)row0; (void)rows; (void)ystage; (void)dst_sp; }
void rl_prepare(const float *src, uint32_t width, int rotate) { (void)src; (void)width; (void)rotate; }
void rl_desc(const struct bns_seq_descriptor *d) { (void)d; }
void rl_engine(const struct engine_op *op) { (void)op; }
int  rl_run(const char *what) { (void)what; return -1; }

int resident_init(void) { resident_state()->ready = 0; return -1; }
int resident_prepare(const float *src, uint32_t width, int rotate)
{
    (void)src; (void)width; (void)rotate; resident_state()->fallbacks++; return -1;
}
int resident_matvec(const struct bons_entry *e, float *dst)
{
    (void)e; (void)dst; resident_state()->fallbacks++; return -1;
}
int resident_silu_init(const float *lut, uint32_t *stage)
{
    (void)stage; resident_state()->silu_lut = lut; return -1;
}
int resident_silu_active(void) { return 0; }
uint32_t resident_silu_patch(uint32_t g_sp, uint32_t m_sp, uint32_t dst_sp, uint32_t fixmap_sp,
                             uint32_t n, uint32_t seg, uint32_t perm_a, uint32_t perm_b)
{
    (void)g_sp; (void)m_sp; (void)dst_sp; (void)fixmap_sp; (void)n; (void)seg; (void)perm_a; (void)perm_b;
    return 0;
}
int resident_ffn_act(const struct bons_entry *gate, const struct bons_entry *up,
                     float *act, float *gate_out, float *up_out)
{
    (void)gate; (void)up; (void)act; (void)gate_out; (void)up_out;
    resident_state()->fallbacks++;
    return -1;
}
int resident_gate_init(const float *sig_lut, uint32_t *image)
{
    (void)sig_lut; (void)image; resident_state()->gate_ready = 0; return -1;
}
int resident_gate_active(void) { return 0; }
int resident_silu_restore(void) { return -1; }
void rl_gate(uint32_t g_sp, uint32_t m_sp, uint32_t fixmap_sp, uint32_t img_sp, uint32_t n)
{
    (void)g_sp; (void)m_sp; (void)fixmap_sp; (void)img_sp; (void)n;
}
uint32_t resident_gate_patch(uint32_t fixmap_sp, const float *g, uint32_t m_sp, float *dst,
                             uint32_t n)
{
    (void)fixmap_sp; (void)g; (void)m_sp; (void)dst; (void)n; return 0;
}

#endif
