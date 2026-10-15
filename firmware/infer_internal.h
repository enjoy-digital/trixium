/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium inference internals
 *
 * What infer.c, infer_float.c, infer_bfp.c and infer_lists.c share: the code layout hints,
 * the float helpers, the scratchpad layout, the channel windows, the model state and the
 * functions one file calls in another.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef INFER_INTERNAL_H
#define INFER_INTERNAL_H

/* Code Layout */
/*-------------*/

// The per-layer hot path is ~28 KB of text against a 32 KB I$ with no L2 behind it (a miss
// is a ~150-cycle HBM round trip). These hints and the cold attributes below move the CPU
// fallback paths out of the hot text (-freorder-blocks-and-partition, see the Makefile):
// same instructions, same numerics.
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define COLD        __attribute__((cold, noinline))

/* Float Helpers */
/*---------------*/

// The LiteX firmware is built with -fno-builtin, so even an RV32F build would call libm for
// every fabsf/lroundf in float->int8 quantization. RISC-V's static RMM rounding mode is
// exactly lroundf's finite-input rule (nearest, ties away from zero); libm stays as the
// bit-identical soft-float fallback.
static inline float f32_abs(float x)
{
    union { float f; uint32_t u; } v = { .f = x };
    v.u &= 0x7fffffff;
    return v.f;
}

static inline uint32_t f32_abs_bits(float x)
{
    union { float f; uint32_t u; } v = { .f = x };
    return v.u & 0x7fffffff;
}

static inline int32_t f32_lround_i32(float x)
{
#if defined(__riscv_flen) && __riscv_flen >= 32
    int32_t r;
    __asm__ volatile("fcvt.w.s %0, %1, rmm" : "=r"(r) : "f"(x));
    return r;
#else
    return (int32_t)lroundf(x);
#endif
}

static inline int32_t f32_q8(float x, float is)
{
    int32_t q = f32_lround_i32(x*is);
    if (q >  127) q =  127;
    if (q < -127) q = -127;
    return q;
}

static inline uint32_t f32_pack_q8x4(const float *src, float is)
{
    // Spell out four independent lanes.  GCC 10 otherwise retains an inner b=0..3 loop
    // with a variable shift, serialising the RV32F conversions and adding four branches
    // to every scratchpad word.
    uint32_t q0 = (uint8_t)(int8_t)f32_q8(src[0], is);
    uint32_t q1 = (uint8_t)(int8_t)f32_q8(src[1], is);
    uint32_t q2 = (uint8_t)(int8_t)f32_q8(src[2], is);
    uint32_t q3 = (uint8_t)(int8_t)f32_q8(src[3], is);
    return q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
}

static inline uint32_t i32_pack_q8x4(const int32_t *src, float is)
{
    // Fused Hadamard boundary: instead of int32 -> FP32*2^e in main RAM, then *2^-e8, both
    // power-of-two scales are combined after the same int32->FP32 conversion, which is
    // bit-identical for the normal finite values produced by the engine.
    uint32_t q0 = (uint8_t)(int8_t)f32_q8((float)src[0], is);
    uint32_t q1 = (uint8_t)(int8_t)f32_q8((float)src[1], is);
    uint32_t q2 = (uint8_t)(int8_t)f32_q8((float)src[2], is);
    uint32_t q3 = (uint8_t)(int8_t)f32_q8((float)src[3], is);
    return q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
}

static inline int f32_e8_from_amax_bits(uint32_t amax)
{
    if (!amax) return 0;
    uint32_t exp = amax >> 23;
    if (exp) return (int)exp - 133; // unbiased exponent + 1 (frexpf), then -7.
    union { uint32_t u; float f; } v = { .u = amax };
    int e8;
    frexpf(v.f, &e8);               // Subnormal activations are not expected; stay exact.
    return e8 - 7;
}

static inline int f32_quant_e8(const float *src, uint32_t n)
{
    uint32_t amax = 0;
    uint32_t i = 0;
    // The reduction is over ordered positive IEEE-754 bit patterns, so four independent
    // loads/comparisons are bit-identical and remove three quarters of the loop control.
    for (; i + 3 < n; i += 4) {
        uint32_t a0 = f32_abs_bits(src[i + 0]), a1 = f32_abs_bits(src[i + 1]);
        uint32_t a2 = f32_abs_bits(src[i + 2]), a3 = f32_abs_bits(src[i + 3]);
        if (a0 <= 0x7f800000u && a0 > amax) amax = a0;
        if (a1 <= 0x7f800000u && a1 > amax) amax = a1;
        if (a2 <= 0x7f800000u && a2 > amax) amax = a2;
        if (a3 <= 0x7f800000u && a3 > amax) amax = a3;
    }
    for (; i < n; i++) {
        uint32_t a = f32_abs_bits(src[i]);
        // Positive finite IEEE-754 bit patterns have the same order as their values.
        // Ignore NaNs as an FP compare would; retain infinity for equivalent overflow.
        if (a <= 0x7f800000u && a > amax) amax = a;
    }
    return f32_e8_from_amax_bits(amax);
}

/* Scratchpad Layout */
/*-------------------*/

// Byte offsets of the resident-BFP regions (int16 mantissas + firmware-tracked exponents;
// sized for the 8B: E=4096, F=12288).
#define SP_X    0x00000 // int8 matvec input (<= 12 KB).
#define SP_Y    0x04000 // Engine int32 results (<= 48 KB).
#define SP_YU   0x10000 // Engine int32 results, second stream (up proj).
#define SP_H    0x1C000 // Normed activations int16.
#define SP_XR   0x1E000 // Resident x (residual stream) int16.
#define SP_G    0x20000 // Gate int16 (silu in/out).
#define SP_U    0x26000 // Up int16.
#define SP_D    0x2C000 // Projection-result int16 (residual addend).
#define SP_FET  0x30000 // Fetch staging (embedding rows, norm16 rows).
#define SP_KVP  0x34000 // KV row-packet staging (built by CPU, engine-stored to HBM).
#define SP_SC   0x36000 // Engine score results (int32, position-major).
#define SP_AW   SP_Y            // Attention: AXPY partials + reduce/norm scratch (free here).
#define SP_AWP8 (SP_Y + 0x2000) // Per-head softmax p8-sum spill beats (32 B each).
#define SP_AWAM (SP_Y + 0x2400) // Per-head reduce-absmax spill beats (32 B each).
#define SP_Q    0x38000 // Resident q int16 (q_dim <= 4096: 8 KB).
#define SP_K    0x3A000 // Resident k int16 (kv_dim <= 1024: 2 KB).
#define SP_V    0x3B000 // Resident v int16.
#define SP_TAB  0x3B800 // Rope table row for the current position ([c14 | s14]).
#define SP_FFSP (SP_TAB + 0x400) // FFN/QKV seq spill beats (5 x 32 B): the tail of the rope
                        // table slot (the row is 2*head_dim bytes, 256 on these models). Not
                        // between SP_H and SP_XR: the 8B's SP_H (8 KB) fills that gap.
#define SP_FET2 0x3C000 // Second norm16 staging slot (k_norm while q_norm sits in SP_FET).
// Resident qk-norm weight rows (qres): head_dim int16 = 256 B per layer per kind, 28 layers =
// 7 KB each, in the tails of the SP_FET and SP_D regions (their live occupants are <= 4 KB).
// Fetched once; the qkr list's rmsnorm_h A1 is patched per layer instead of two engine
// fetches per layer.
#define SP_QNW  (SP_FET + 0x1000)
#define SP_KNW  (SP_D + 0x1000)

// Fixed exponent for CPU-produced normed vectors (values bounded ~16 by construction).
#define E_NORM  (-11)

/* Heap and Channel Windows */
/*--------------------------*/

// Heap: after the BONS header/table area. The CPU's RAM is HBM pseudo-channel 4 and engine
// channel 3's weight window overlaps it at [64 MB, ~141 MB) (see layout.py HBM_WEIGHT_BASE):
// the low heap must stay under 64 MB, and anything big goes to the HIGH heap above the
// weights+KV tail. The low heap holds a few MB of norms and tables; the only big
// consumer is the CPU-side KV cache, which is a debug fallback (HW attention keeps its own
// KV in HBM at full context) and is capped at CPU_KV_CTX positions to fit.
#define HEAP_BASE    (TRIXIUM_HDR_ADDR + 0x100000)
#define CPU_KV_CTX   512

// Engine-side addresses vs the CPU's. On the FK33 the CPU window passes through 1:1 (main_ram
// at 0x4000_0000 IS HBM address 0x4000_0000 on port 0, inside pseudo-channel 4 = engine
// channel 3's window). On a board with several memory controllers (the XCU1525's four DDR4
// channels) engine channel c lives on controller c / CHAN_MOD at that controller's local
// BASE + (c % CHAN_MOD)*STRIDE, and main_ram is controller 0's local [MEM_CPU_OFFSET, +size).
// Two SoC constants describe both; absent, the FK33's geometry.
#ifndef TRIXIUM_CHAN_MOD
#define TRIXIUM_CHAN_MOD 0                       // 0: one flat engine address space.
#endif
#ifndef TRIXIUM_MEM_CPU_OFFSET
#define TRIXIUM_MEM_CPU_OFFSET 0x40000000ull      // Engine-side address of main_ram's origin.
#endif
#ifndef TRIXIUM_CHAN_SKEW
#define TRIXIUM_CHAN_SKEW 0                       // Per-channel bank skew inside a controller.
#endif
static inline uint64_t chan_window(uint32_t c)  // Engine-side start of channel c's window.
{
    uint32_t local = TRIXIUM_CHAN_MOD ? (c % TRIXIUM_CHAN_MOD) : c;
    return (uint64_t)TRIXIUM_CHAN_WEIGHT_BASE + (uint64_t)local*TRIXIUM_CHAN_STRIDE
           + (uint64_t)local*TRIXIUM_CHAN_SKEW;
}
static inline int chan_on_cpu_memory(uint32_t c) // Same controller as main_ram?
{
    return TRIXIUM_CHAN_MOD ? (c / TRIXIUM_CHAN_MOD) == 0 : 1;
}
static inline uint32_t cpu_addr_of(uint64_t eng) // Engine-side -> CPU address (main_ram).
{
    return (uint32_t)(0x40000000ull + (eng - TRIXIUM_MEM_CPU_OFFSET));
}

/* Shared Layout */
/*---------------*/

// Score padding granule: softmax8's x2 unroll (ISA rev 2 microcode) walks two 16-entry
// blocks per iteration and has no odd-tail form, so scores are padded to a 32-multiple
// with INT32_MIN there (those entries emit p8 = 0 - the sum and the AXPY do not change).
#if TRIXIUM_PROGS_VREGS >= 16
#define SC_PAD 32
#else
#define SC_PAD 16
#endif
#define SC_PADN(cnt) ((SC_PAD - ((cnt) & (SC_PAD - 1))) & (SC_PAD - 1))

#define KVP_SLOTS 16                    // KV row-packet staging slots (infer_bfp.c).
#define NRM_MAXL  64                    // Layers of the norm16 entry cache.
#define SP_QKSP   (SP_FFSP + 64)        // QKV list absmax spills (infer_lists.c).
#define SP_LUTQ   (SP_DESC + 188*64)    // 32 x int32 eps for the q norm (descriptor slots
#define SP_LUTK   (SP_LUTQ + 128)       // 188-191, past every list).
#define SP_SCB2   sp_scb2               // Second attention scores buffer (0 = none).

// seq_eps_int cache, index es+64; 0 = not computed yet. int32 over [-64, 63]: the qkvx LUT
// fill asks for 32 es values per norm per layer (including the big-eps, very negative es
// entries) and a miss is soft-float work. In the main_ram static window (not SRAM .bss,
// which is at its 4 KiB stack floor), so NOT zero-initialized: infer_static_init() clears
// it.
#define eps_memo ((int32_t *)TRIXIUM_EPS_MEMO_ADDR)

/* Scratchpad Access */
/*-------------------*/

static inline volatile int16_t *sp16(uint32_t off) { return (volatile int16_t *)(SCRATCHPAD_BASE + off); }
static inline volatile int32_t *sp32(uint32_t off) { return (volatile int32_t *)(SCRATCHPAD_BASE + off); }
static inline volatile int8_t  *sp8(uint32_t off)  { return (volatile int8_t  *)(SCRATCHPAD_BASE + off); }

#ifdef TRIXIUM_HAS_ZBB
// Zbb clz. GCC 10.1 knows no Zb*, so the encoding is spelled out:
// .insn i 0x13, 1, rd, rs1, 0x600 == clz rd, rs1 (0x60059513 for a0, a1). Defined for 0
// (returns 32), so no zero test is needed.
static inline __attribute__((always_inline)) int bitlen32(uint32_t x)
{
    uint32_t r;
    __asm__ (".insn i 0x13, 1, %0, %1, 0x600" : "=r"(r) : "r"(x));
    return 32 - (int)r;
}
#else
static inline __attribute__((always_inline)) int bitlen32(uint32_t x)
{
    return x ? 32 - __builtin_clz(x) : 0;
}
#endif

/* infer.c */
/*---------*/

extern const struct bons_header *H;
extern int n_ch, head_dim, n_head, n_kv_head, n_embd, ffn_dim, n_layer;
extern uint8_t *heap, *heap_hi;
extern uint32_t kv_rb, kv_region;
extern int8_t *kv_ekmax;

void *halloc(uint32_t n);
void *halloc_hi(uint32_t n);
float fp16_to_float(uint16_t b);
COLD int quantize_f_pow2(const float *src, int n, uint32_t sp_off);
int kv_ek_idx(int l, int kvh, int kind);
uint32_t kv_region_addr(int l, int kvh, int kind);
void kv_scores_br(int l, int kvh, int kind, int *B, int *R);

/* infer_float.c */
/*---------------*/

COLD void matvec_rows_f(const struct bons_entry *e, uint32_t row0, uint32_t rows,
                        float xscale, float *dst, int32_t *acc, int acc_add, int dma_result);
COLD void embed_row_f(const struct bons_entry *e, int row, float *dst);
COLD void rope_apply(float *vec, int n_heads_v, int pos);
void rope_init(void);

/* infer_bfp.c */
/*-------------*/

extern const struct bons_entry *emb16_ent, *emb16x_ent;
extern int kvp_scale_e8[KVP_SLOTS];
extern int rq8_prefetched;

// The int16 norm rows carry a per-tensor exponent in the entry's bias byte; the rmsnorm
// program's A6 is the combined shift.
static inline int w16_shift(const struct bons_entry *e)
{
    return (int)(int8_t)e->bias - E_NORM - 15;
}

void eps_init(void);
int quant8_shift(uint32_t amax);
uint32_t norm_fetch_submit(const struct bons_entry *w16);
int resid_add_pf(uint32_t src, int esrc, uint32_t d, int ed, int n, uint32_t dmax,
                 uint32_t *xamax, const struct bons_entry *next_w16);
int op_rmsnorm_q8_sp(uint32_t src, int esrc, const float *w, const struct bons_entry *w16,
                     int n, uint32_t dst16, uint32_t dst8);
int op_silu_q8_sp(uint32_t gsrc, int eg, uint32_t usrc, int eu, int n, uint32_t dst8);
#define RESID_TAIL_OFF (-1000) // resid_norm_q8 ran nothing (no quant8 shift is ever this).
int resid_norm_q8(const struct engine_op *pf, uint32_t src, int eproj, int *eres, uint32_t *xamax,
                  const struct bons_entry *w16, int n, int fetched);
int op_norm_i32_sp_a(uint32_t src, uint32_t dst, int n, uint32_t amax, uint32_t *omax);
int op_norm_i32_sp_m(uint32_t src, uint32_t dst, int n, uint32_t *omax);
static inline int op_norm_i32_sp(uint32_t src, uint32_t dst, int n)
{
    return op_norm_i32_sp_m(src, dst, n, 0);
}
int attn_all_hw(void);
void vec_quant8_head(uint32_t sp, uint32_t dst, int n);
uint32_t kvp_slot(int kvh, int kind);
void kv_pack_prep_res(int l, int kvh, int kind, int es);
void kv_pack_prep_layer(int l, int ev);
int embed_row_fetch(const struct bons_entry *e, int row, uint32_t sp_dst);

/* infer_lists.c */
/*---------------*/

extern int seq_qkr_count, seq_qkr_di_qn, seq_qkr_di_kn;
extern int qkvx_ok, qkv_prepped, qkn_resident;
extern int seq_qkv_base, seq_qkv_len, seq_qkv_di_kn;
extern int kv_pipe_active, kv_mstore_active;
extern uint8_t seq_kvst_di[32];
extern uint32_t kvst_rb_set, kvst_chan_set;
extern int seq_attn_base, seq_attn_dph, seq_attn_heads, attn_pipe_active, attn_prepatched;
extern uint32_t sp_scb2, sc_b_cap_bytes;
extern int seq_ffn_base, seq_ffn_len, ffn_di_a4, ffn_prepatched;
extern int seq_scale8_base;
extern uint64_t *const fprof_acc_list, *const fprof_acc_ops;
extern uint32_t *const fprof_steps, *const fprof_ts;
extern uint32_t fprof_nlist, fprof_nops;
extern int fprof_nsteps, fprof_nts;

void seq_build_qkr(void);
void seq_build_attn(void);
void seq_build_ffn(void);
void seq_build_scale8(void);
void seq_build_qkv(void);
int seq_eps_int(int es);
void attn_list_prep_token(int pos);
void attn_patch_slots(int l, int h0, int nh, int chunked);
void ffn_prep_layer(int l, const struct bons_entry *next_w16);
void qkv_prep_layer(int l);

// List-relative dcount ("descriptors completed") threshold after which head j's fold
// inputs (softmax8's AWP8 self-spill and spill(j)'s AWAM) are certainly written: one
// past norm(j), whose producers are all earlier vec descriptors in the same list.
static inline uint32_t attn_fold_ready(int j, int chunk)
{
    if (!attn_pipe_active) return (uint32_t)(j*5 + 5);
    if (j >= chunk - 2)    return (uint32_t)(1 + (j + 1)*5 + 2); // last-group/epilogue slots.
    return (uint32_t)(1 + (j + 1)*5 + 3);
}

static inline int attn_list_len(int nh)
{
    return nh*seq_attn_dph + (attn_pipe_active ? 3 : 0); // Prologue + 2-slot epilogue.
}

#endif /* INFER_INTERNAL_H */
