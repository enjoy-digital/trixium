/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium hybrid models forward (Bonsai 2: Gated DeltaNet + gated attention + rotated basis)
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef HYBRID_H
#define HYBRID_H

/* Model */
/*-------*/

int  hybrid_present(void);                       // The loaded image is a hybrid (BONS v3 fields).
int  hybrid_init(void);                          // After the header checks; 0 on success.
void hybrid_reset(void);                         // New generation: states and position to zero.
int  hybrid_pos(void);                           // Positions processed since the reset.
int  hybrid_context_capacity(void);              // Actual KV allocation after heap-fit reduction.
int  hybrid_context_unbounded(void);             // Past the capacity the KV ring goes on.
void hybrid_set_pin(int n);                      // The next reset's KV positions kept past the window.
int  hybrid_forward(int token, int want_logits); // Next token (argmax) or -1.
// Batched prefill (engines with the multi-row matvec): two prompt tokens without logits in one
// layer-major pass, bit-identical to two hybrid_forward(t, 0). 0 done, -1 failed, -2 not run
// (no hardware, `prefill2 off`, the model does not fit the lists, the context cap).
int  hybrid_forward2(const int *tokens);
int  hybrid_forward2_ok(void);                   // hybrid_forward2 would run now
void hybrid_prefill2_enable(int on);             // Console `prefill2 on|off|ffn` (1, 0, 2: the FFN only; default on).
int  hybrid_prefill2_enabled(void);               // 1 on, 2 the FFN only, 0 off.
// Two conversations decoded together (the same engines): sequence 1's state set allocated on
// the first call (0 ok, -1 not built or does not fit), the sequence the other calls act on
// (0 everywhere else), one step of both: out[s] = the next token of sequence s after tokens[s],
// bit-identical to hybrid_forward(tokens[s], 1) on each. 0 done, -1 failed, -2 not run.
int  hybrid_dual_setup(void);
void hybrid_seq_select(int s);
int  hybrid_seq_pos(int s);                      // Sequence s's position.
void hybrid_lmhead2_enable(int on);              // Console `gen2 lmhead on|off`: both LM heads in one pass (default on).
int  hybrid_lmhead2_enabled(void);
int  hybrid_forward_dual(const int *tokens, int *out);
int  hybrid_ckpt_save(void);                     // 1: the next forward can be re-run ...
void hybrid_ckpt_restore(void);                  // ... from here (ECC retry).

/* Profiling */
/*-----------*/

void hybrid_stats_reset(void);                   // Reset the per-stage cycle counters.
void hybrid_stats_print(void);                   // Print the per-stage averages (stats' hybprof lines).

/* Console Controls */
/*------------------*/

void hybrid_dnres_enable(int on);                // Console `dnres`: the resident DeltaNet layer.
int  hybrid_dnres_enabled(void);
void hybrid_dnstream_enable(int on);             // Console `dnstream`: op 5 heads (at next reset).
int  hybrid_dnstream_enabled(void);
void hybrid_attnstream_enable(int on);           // Console `attnstream`: op 6 (at next reset).
int  hybrid_attnstream_enabled(void);
void hybrid_lm_chunk(uint32_t rows);             // Console `lmchunk <rows>` (0 = automatic).
void hybrid_hwargmax_enable(int on);             // Console `hwargmax`: the LM head list's argmax.
int  hybrid_hwargmax_enabled(void);

#endif /* HYBRID_H */
