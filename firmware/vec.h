// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// BonsaiVectorUnit driver: program upload at init, one VECRUN at a time.

#ifndef VEC_H
#define VEC_H

#include <stdint.h>

#ifdef CSR_VECUNIT_BASE
void vec_init(void); // Upload the assembled programs (progs.h) into the instruction BRAM.
// Run a program at `pc` with up to 8 args; returns S6 (the conventional result scalar).
uint64_t vec_run(uint16_t pc, const int32_t *args, int n_args);
#endif

#endif
