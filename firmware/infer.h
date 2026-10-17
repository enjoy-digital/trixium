/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium Qwen3 token loop and the float matvec primitives of the hybrid path
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef INFER_H
#define INFER_H

#include <stdint.h>

/* Generation */
/*------------*/

int  infer_init(void);   // Load norms etc; returns 0 on success.
// Feed prompt ids then generate up to max_new tokens (greedy unless sampling is set); calls
// emit() per new token. Returns the number of generated tokens, or -1 on error.
int  infer_generate(const int *prompt, int n_prompt, int max_new, void (*emit)(int tok));
// Same, continuing a conversation: buf[0..resident) are in the KV cache from the previous
// call (check infer_gen_serial() has not moved), the rest is new. resident 0 = fresh.
int  infer_generate_from(const int *buf, int n, int resident, int max_new, void (*emit)(int tok));
int  infer_kv_resident(void);       // Tokens in the KV cache after a generation (= n_pos).
void infer_kv_fill(uint32_t pattern); // Diagnostic (`kvfill`): every channel's KV region <- pattern.
uint32_t infer_gen_serial(void);    // Bumped by every generation.
int  infer_context_capacity(void); // Actual allocation, not the image's requested max_ctx.
struct infer_prefill_profile {
    uint64_t reset_cycles, prefill_cycles;
    uint32_t tokens, forwards;
    int valid; // Only set after the complete prompt was successfully processed.
};
const struct infer_prefill_profile *infer_last_prefill(void);
// Stop set for infer_generate: a generated token in it is emitted and ends the request
// (the chat tools pass <|im_end|>; the tokenizer lives on the host). n = 0 clears it.
#define INFER_STOP_MAX 4
void infer_set_stop(const int *ids, int n);
void infer_static_init(void);   // Clear the main_ram static-window state infer.c keeps (boot).

/* Sampling */
/*----------*/

// Temperature (0 = greedy), top-p, repetition penalty (1 = none), PRNG seed (0 = keep).
// infer_set_recent names the token buffer the penalty looks back into.
void infer_set_sampling(float temp, float top_p, float rep, uint32_t seed);
void infer_get_sampling(float *temp, float *top_p, float *rep);
void infer_set_recent(const int *buf, const int *n_ptr);

/* Sequencer Lists and Debug */
/*---------------------------*/

// Sequencer lists (console `seq`): on by default. The same ops run CPU-dispatched per head
// when off.
extern int seq_use;

void infer_dbg(int n);    // Checksum the resident intermediates of the next n forwards (dbg).
void infer_dbg_f32(const char *tag, int layer, const float *values, uint32_t n);
int  infer_dbg_armed(void);   // Typed traces on: callers that skip a main-RAM copy must not.
void infer_attn_chunk(int cap); // Debug: cap the heads per attention list (attnchunk N).
extern uint32_t infer_forwards;  // Forwards run since boot.
extern uint32_t infer_ecc_retries, infer_ecc_unrecovered; // DDR ECC: forwards re-run / still bad.
extern uint32_t infer_ecc_force;                          // Console `eccforce N`: re-run the next N.
void infer_qkvx_report(void);     // Print the qkvx state (at init; never mid-generation).
void infer_seq(int on);           // Sequencer lists on/off (seq); rebuilds the lists.

// FFN profiling (console `fprof`).
extern int fprof_use;
void fprof_reset(void);
void fprof_print(void);

/* Hybrid Path Primitives */
/*------------------------*/

// The hybrid CPU path's primitives (hybrid.c): the engine matvec on a float vector (row
// order out), main_ram allocation (big = the high heap when there is one), the room left,
// an F32 tensor fetch, the embedding row as floats.
struct bons_entry;
uint32_t infer_matvec_cols(const struct bons_entry *e);   // K of the whole matvec (K-segments summed).
void     infer_matvec_f(const struct bons_entry *e, const float *src, float *dst);
int      infer_quantize_f(const float *src, uint32_t n);  // Put Q8 input in SP_X; return exponent.
void     infer_matvec_f_quantized(const struct bons_entry *e, int e8, float *dst);
// Prism folded-basis fast path: rotate blockwise into retained int32+exponent form, pack
// the exact downstream Q8 bytes without materializing FP32, then run a possibly K-split
// projection. Widths that fit SP_X use infer_matvec_f_quantized after the first call.
#define INFER_ROTATE_UNSUPPORTED 0x40000000
int      infer_rotate_quantize_f(const float *src, uint32_t width);
void     infer_matvec_f_rotated_quantized(const struct bons_entry *e, int e8, float *dst);
void    *infer_halloc(uint64_t n, int big);
uint64_t infer_heap_room(int big);
void     infer_fetch_f32(uint32_t addr, uint32_t beats, float *dst, int n);
int      infer_embed_f(int token, float *dst);

#endif
