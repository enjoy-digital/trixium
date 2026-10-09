/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium per-op CPU/HW dispatch
 *
 * Bit i of hwops_mask selects the hardware implementation of perf_op i (see perf.h); the mask
 * is always AND-ed with hw_caps() (the ops the firmware has a hardware path for), so requesting
 * HW for any other op stays on the CPU. The CPU implementations remain functional references.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef OPS_H
#define OPS_H

#include <stdint.h>

#include "perf.h"

extern uint32_t hwops_mask;

uint32_t hwops_default(void); // Every op with a hardware path (hw_caps()).
void hwops_init(void);        // Apply the default mask (call once at boot).

uint32_t    hw_caps(void);    // The ops with a hardware path (constant).
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
