// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause
//
// Per-op CPU/HW dispatch. Bit i of hwops_mask selects the hardware implementation of
// perf_op i (see perf.h); the effective mask is always AND-ed with hw_caps() (gateware
// capability constants), so requesting HW for an absent unit silently stays on the CPU.
// The CPU implementations remain fully functional forever.

#ifndef OPS_H
#define OPS_H

#include <stdint.h>

#include "perf.h"

extern uint32_t hwops_mask;

uint32_t hwops_default(void); // Caps minus the sequencer attention path (see ops.c).
void hwops_init(void);        // Apply the default mask (call once at boot).

uint32_t    hw_caps(void);                  // Which ops this gateware can do in HW.
static inline int op_on_hw(enum perf_op op)
{
    extern uint32_t hwops_mask;
    return (hwops_mask >> op) & 1;
}
void        hwops_set(enum perf_op op, int hw); // Respects hw_caps().
void        hwops_set_all(int hw);
void        hwops_print(void);
const char *op_name(enum perf_op op);

#endif
