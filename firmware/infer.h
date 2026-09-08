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
// Sequencer attention lists: off by default (they wedge the engine on this gateware - see
// infer.c). The same ops run CPU-dispatched per head when off.
extern int seq_use;

void infer_vbench(void);  // Stage-by-stage cycle microbenchmark (run after hdrload).
void infer_attn_pipe(int on);   // Pipelined attention list on/off (aovl; rebuilds lists).
void infer_kv_pipe(int on);     // In-list NOWAIT KV stores on/off (kovl; rebuilds lists).
int  infer_kv_pipe_state(void);
void infer_kvq_redundant(int on); // Redundant per-slot KV-pack quant8s (kvq; rebuilds lists).
extern uint32_t infer_forwards;  // Forwards run since boot.
void infer_kv_mstore(int on);     // In-list KV multi-store vs per-slot stores (kvms).
int  infer_kv_mstore_state(void);
void infer_ffn_split(int on);     // FFN SiLU split: gate half under the up stream (fsplit).
int  infer_ffn_split_state(void);
int  infer_kvq_redundant_state(void);
int  infer_attn_pipe_state(void);
void infer_spec(int on);                    // Speculative decode on/off (lossless, greedy).
void infer_spec_stats(int *acc, int *rej);  // Accepted / rejected draft counts.

#endif
