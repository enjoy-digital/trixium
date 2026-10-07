// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// The hybrid models (Bonsai 2: Gated DeltaNet + gated attention + the rotated basis) on the
// CPU path - see hybrid.c.
#ifndef HYBRID_H
#define HYBRID_H
int  hybrid_present(void);                  // The loaded image is a hybrid (BONS v3 fields).
int  hybrid_init(void);                     // After the header checks; 0 on success.
void hybrid_reset(void);                    // New generation: states and position to zero.
void hybrid_stats_reset(void);              // Reset exact-recurrence stage cycle counters.
void hybrid_stats_print(void);              // Print averages when the recurrence HW is active.
void hybrid_dnres_enable(int on);             // Console `dnres`: the resident DeltaNet layer.
int  hybrid_dnres_enabled(void);
void hybrid_dnstream_enable(int on);          // Console `dnstream`: op 5 heads (at next reset).
int  hybrid_dnstream_enabled(void);
void hybrid_attnstream_enable(int on);        // Console `attnstream`: op 6 (at next reset).
int  hybrid_attnstream_enabled(void);
void hybrid_attn_fail_at(int p);                // Console `attnfail <pos>` (debug).
void hybrid_prefetch_enable(int on);            // Console `prefetch`: run-ahead weight reads.
int  hybrid_prefetch_enabled(void);
void hybrid_lm_check(void);                   // Console `lmcheck` (debug).
void hybrid_lm_chunk(uint32_t rows);          // Console `lmchunk <rows>` (0 = automatic).
void hybrid_hwargmax_enable(int on);          // Console `hwargmax`: the LM head list's argmax.
int  hybrid_hwargmax_enabled(void);
void hybrid_xres_enable(int on);              // Console `xres`: the resident residual stream.
int  hybrid_xres_enabled(void);
int  hybrid_pos(void);                      // Positions processed since the reset.
int  hybrid_context_capacity(void);         // Actual KV allocation after heap-fit reduction.
int  hybrid_forward(int token, int want_logits); // Next token (argmax) or -1.
// The MTP head (speculative decoding): one draft at position p from the token there and the
// hidden state that predicted it; the loop with sequential verification (stats: passes,
// drafted, accepted, head cycles).
int  hybrid_mtp_present(void);
int  hybrid_mtp(int token, int p, const float *hprev, float *hout);
int  hybrid_mtp_generate(const int *prompt, int n_prompt, int n_new, int k, int batched,
                         void (*emit)(int tok), uint64_t *stats);
int  hybrid_spec_max(void);              // forwards per speculative pass (state versions - 1)
// Batched speculative forwards (engine cmd.nb, BONSAI_HAS_NBATCH): console `nbatch on|off`.
int  hybrid_forward_batch(const int *tok, int nb, int *ys, float *hs);
void hybrid_nbatch_enable(int on);
void hybrid_mtpdbg_enable(int on);     // console `mtpdbg on|off`
int  hybrid_nbatch_enabled(void);
void hybrid_nbatch_stats(uint32_t *batched, uint32_t *declined);   // FFN layers since init
int  hybrid_spec_begin(void);
int  hybrid_spec_end(int keep);
#define hybrid_mtp_oracle ((int *)BONSAI_MTP_ORACLE_ADDR)  // draft overrides by position (tests)
extern int hybrid_mtp_noracle;
extern int hybrid_mtp_drafts[64], hybrid_mtp_ndrafts;  // the last generation's first drafts
#endif
