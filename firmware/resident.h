// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// Resident FP32 projections for the hybrid (27B) path: the activation quantization, the
// integer Hadamard rotation, the whole-vector requantization and the int32 -> FP32 result
// restoration run as sequencer descriptor lists inside the scratchpad (units 4, 3, 5 and 8
// around the legacy engine matvec, resident FETCH/STORE for the FP32 vectors). The CPU
// builds descriptors and invalidates its cache; it touches no element. Numerically this is
// the fused CPU path of infer.c (same exponent rule, ties-away Q8, int32 K-segment sum, RNE
// int32->FP32 then a power-of-two scale), so the typed-checkpoint gate must stay 87/87.
//
// Needs a bitstream with BONSAI_HAS_HADAMARD, BONSAI_HAS_ACTIVATION_QUANTIZE (ABI 1.1) and
// BONSAI_HAS_PROJECTION_RESTORE (v11+). On any other bitstream resident_available() is 0
// and hybrid.c keeps its existing path; a failed list disables the path for the session
// after an engine soft reset (the weights stay).
#ifndef RESIDENT_H
#define RESIDENT_H

#include <stdint.h>
#include "bons.h"

int  resident_available(void);   // The bitstream carries units 3/4/5/8 with the expected ABIs.
int  resident_init(void);        // Extract the rotation sign bits from the image. 0 = ready.
void resident_enable(int on);    // Console A/B: default on when available.
int  resident_active(void);      // available && enabled && not disabled by a failure.

// Quantize src[width] (FP32, main RAM) into the resident Q8 vector, rotating it first when
// `rotate` (width must be a multiple of the Hadamard block). Leaves the whole-vector
// exponent resident too. 0 = ok, -1 = unsupported/failed (caller falls back).
int  resident_prepare(const float *src, uint32_t width, int rotate);
// Multiply the resident Q8 vector by tensor e (K-segments summed in int32), restore FP32 and
// DMA the rows into dst (main RAM, 32-byte aligned, rows*4 bytes). 0 = ok, -1 = unsupported.
int  resident_matvec(const struct bons_entry *e, float *dst);

// State and per-forward accounting (cycles include the descriptor build and the CPU-side
// cache work). Lives in the main_ram static window (static_memory.h), not SRAM .bss: the
// 8 KiB SRAM is at its 4 KiB stack floor. Initialized by resident_init(); the flags first.
struct resident_state {
    uint64_t prepare_cycles, matvec_cycles;
    uint32_t prepares, matvecs, fallbacks;
    uint32_t sign_base[4];       // Scratchpad offset of each Hadamard width's sign bits, 0 = none.
    const float *prep_src;       // hybrid.c: the last prepared source, for a CPU fallback.
    int32_t prep_width, prep_rotate;
    uint8_t enabled, disabled, ready, n_ch;
};
struct resident_state *resident_state(void);
void resident_stats_reset(void);

#endif
