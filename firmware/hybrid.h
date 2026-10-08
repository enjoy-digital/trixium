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
int  hybrid_forward(int token, int want_logits); // Next token (argmax) or -1.

/* Profiling */
/*-----------*/

void hybrid_stats_reset(void);                   // Reset the per-stage cycle counters.
void hybrid_stats_print(void);                   // Print the per-stage averages (hybprof).

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
