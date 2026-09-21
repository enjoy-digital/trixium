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
int  hybrid_pos(void);                      // Positions processed since the reset.
int  hybrid_context_capacity(void);         // Actual KV allocation after heap-fit reduction.
int  hybrid_forward(int token, int want_logits); // Next token (argmax) or -1.
#endif
