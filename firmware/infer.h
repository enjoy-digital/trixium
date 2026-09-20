// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

#ifndef INFER_H
#define INFER_H

#include <stdint.h>

int  infer_init(void);   // Load norms etc; returns 0 on success.
// Feed prompt ids then generate up to max_new tokens (greedy); calls emit() per new token.
// Returns the number of generated tokens.
int  infer_generate(const int *prompt, int n_prompt, int max_new, void (*emit)(int tok));
// Same, continuing a conversation: buf[0..resident) are in the KV cache from the previous
// call (check infer_gen_serial() has not moved), the rest is new. resident 0 = fresh.
int  infer_generate_from(const int *buf, int n, int resident, int max_new, void (*emit)(int tok));
int  infer_kv_resident(void);       // Tokens in the KV cache after a generation (= n_pos).
uint32_t infer_gen_serial(void);    // Bumped by every generation.
// Stop set for infer_generate: a generated token in it is emitted and ends the request
// (the chat tools pass <|im_end|>; the tokenizer lives on the host). n = 0 clears it.
#define INFER_STOP_MAX 4
void infer_set_stop(const int *ids, int n);
// Sampling: temperature (0 = greedy), top-p, repetition penalty (1 = none), PRNG seed
// (0 = keep). infer_set_recent names the token buffer the penalty looks back into.
void infer_set_sampling(float temp, float top_p, float rep, uint32_t seed);
void infer_get_sampling(float *temp, float *top_p, float *rep);
void infer_set_recent(const int *buf, const int *n_ptr);
// Sequencer attention lists: off by default (they wedge the engine on this gateware - see
// infer.c). The same ops run CPU-dispatched per head when off.
extern int seq_use;

void infer_vbench(void);  // Stage-by-stage cycle microbenchmark (run after hdrload).
void infer_dbg(int n);    // Checksum the resident intermediates of the next n forwards (dbg).
void infer_dbg_dump(const char *tag, int l, int pos); // ...and print one of them whole.
void infer_attn_pipe(int on);   // Pipelined attention list on/off (aovl; rebuilds lists).
void infer_attn_chunk(int cap); // Debug: cap the heads per attention list (attnchunk N).
void infer_kv_pipe(int on);     // In-list NOWAIT KV stores on/off (kovl; rebuilds lists).
int  infer_kv_pipe_state(void);
void infer_kvq_redundant(int on); // Redundant per-slot KV-pack quant8s (kvq; rebuilds lists).
extern uint32_t infer_forwards;  // Forwards run since boot.
void infer_kv_mstore(int on);     // In-list KV multi-store vs per-slot stores (kvms).
void infer_qkr_reorder(int on);   // qkr list: Q chain behind the store (qkro).
void infer_qfix(int on);          // Scan-free fixed-shift Q quant8 in the qkr list (qfix).
void infer_embfetch(int on);      // Embedding row fetched pre-dequantized from the image (embf).
void infer_qkvx(int on);          // K chain + V pack inside the qkv list, eps from a LUT (qkvx).
void infer_qkvx_report(void);     // Print the qkvx state (init and the toggle; never mid-generation).
void infer_qkvl(int on);          // QKV projections + norms as one list (qkvl); rebuilds the lists.
void infer_seq(int on);           // Sequencer lists on/off (seq); rebuilds the lists.
int  infer_qkvx_state(void);
int  infer_embfetch_state(void);
int  infer_qfix_state(void);
int  infer_qkr_reorder_state(void);
int  infer_kv_mstore_state(void);
void infer_ffn_split(int on);     // FFN SiLU split: gate half under the up stream (fsplit).
int  infer_ffn_split_state(void);
int  infer_kvq_redundant_state(void);
int  infer_attn_pipe_state(void);
void infer_spec(int on);                    // Speculative decode on/off (lossless, greedy).
void infer_spec_stats(int *acc, int *rej);  // Accepted / rejected draft counts.

// The hybrid CPU path's primitives (hybrid.c): the engine matvec on a float vector (row
// order out), main_ram allocation (big = the high heap when there is one), the room left,
// an F32 tensor fetch, the embedding row as floats.
struct bons_entry;
uint32_t infer_matvec_cols(const struct bons_entry *e);   // K of the whole matvec (K-segments summed).
void     infer_matvec_f(const struct bons_entry *e, const float *src, float *dst);
void    *infer_halloc(uint64_t n, int big);
uint64_t infer_heap_room(int big);
void     infer_fetch_f32(uint32_t addr, uint32_t beats, float *dst, int n);
int      infer_embed_f(int token, float *dst);
#endif
