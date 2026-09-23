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

// Quantize src[width] (FP32, main RAM; NULL = the activation resident_ffn_act left in the
// scratchpad) into the resident Q8 vector, rotating it first when
// `rotate` (width must be a multiple of the Hadamard block). Leaves the whole-vector
// exponent resident too. 0 = ok, -1 = unsupported/failed (caller falls back).
int  resident_prepare(const float *src, uint32_t width, int rotate);
// Multiply the resident Q8 vector by tensor e (K-segments summed in int32), restore FP32 and
// DMA the rows into dst (main RAM, 32-byte aligned, rows*4 bytes). 0 = ok, -1 = unsupported.
int  resident_matvec(const struct bons_entry *e, float *dst);

// Unit 11 (BONSAI_HAS_SILU): load the firmware's SiLU table into the unit (after
// resident_init), through `stage` (main RAM, 32-byte aligned, 1,025*32 bytes, free after
// the call), and self-test it. 0 = the FFN activation can run resident.
int  resident_silu_init(const float *lut, uint32_t *stage);
int  resident_silu_active(void);
void resident_silu_enable(int on);   // Console A/B.
// With the FFN input prepared (resident_prepare), compute act = up * silu(gate) in the
// scratchpad (bit-exact with hybrid.c's silu_mul_fv; the rare out-of-range elements are
// recomputed on the CPU) and leave it in RS_F for resident_prepare(NULL, ...). act also
// receives a copy in main RAM (the fallback source); gate_out/up_out, when not NULL,
// receive the projections for traces. All 32-byte aligned. 0 = ok, -1 = declined/failed.
int  resident_ffn_act(const struct bons_entry *gate, const struct bons_entry *up,
                      float *act, float *gate_out, float *up_out);

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
    uint64_t silu_cycles;
    uint32_t silu_lists, silu_fixups;
    const float *silu_lut;
    uint8_t silu_enabled, silu_ready, act_resident;
};
struct resident_state *resident_state(void);
void resident_stats_reset(void);

#endif
