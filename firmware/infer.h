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
void infer_spec(int on);                    // Speculative decode on/off (lossless, greedy).
void infer_spec_stats(int *acc, int *rej);  // Accepted / rejected draft counts.

#endif
