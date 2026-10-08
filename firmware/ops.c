/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium per-op CPU/HW dispatch
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include <stdio.h>

#include <generated/csr.h>
#include <generated/soc.h>

#include "ops.h"

/* Variables */
/*-----------*/

// Every op the gateware implements, on by default: each one is gated bit-exact against the
// interpreter (VECCRC) and token-gated in simulation, and the CPU paths are debug fallbacks.
// hwops_init() is called after the model header is known; `hwops all cpu` still selects the
// fallback path for A/B work.
uint32_t hwops_mask;

static const char *names[OP_COUNT] = {
    "embed", "rmsnorm", "qknorm", "rope", "kvquant", "scores", "softmax", "wv",
    "quant8", "norm_i32", "silu_mul", "resid_add", "lmhead", "engine_wait",
};

/* Mask */
/*------*/

// All or nothing: the resident loop keeps activations in the scratchpad, so a mixed mask pays a
// float round-trip every layer and ends up slower than either all-HW or all-CPU.
uint32_t hwops_default(void)
{
    return hw_caps();
}

void hwops_init(void)
{
    hwops_mask = hwops_default();
}

const char *op_name(enum perf_op op)
{
    return (op < OP_COUNT) ? names[op] : "?";
}

uint32_t hw_caps(void)
{
    uint32_t caps = 0;
    caps |= (1 << OP_SILU_MUL) | (1 << OP_QUANT8) | (1 << OP_NORM_I32) | (1 << OP_RESID_ADD);
    caps |= (1 << OP_RMSNORM);
    // KVQUANT/SCORES/WV all ride the KV regions + engine (AXPY is in the base engine).
    caps |= (1 << OP_KVQUANT) | (1 << OP_SCORES) | (1 << OP_WV);
    caps |= (1 << OP_LMHEAD);
    // Fused resident attention: softmax8 + AXPY + reduce, and the resident front
    // (qk-norm/rope on the vector unit, image rope tables, resident kv-pack).
    caps |= (1 << OP_SOFTMAX) | (1 << OP_QKNORM) | (1 << OP_ROPE);
    return caps;
}

void hwops_set(enum perf_op op, int hw)
{
    uint32_t bit = 1u << op;
    if (hw) {
        if (!(hw_caps() & bit)) {
            printf("hwops: %s has no hardware implementation in this gateware\n", op_name(op));
            return;
        }
        hwops_mask |= bit;
    } else {
        hwops_mask &= ~bit;
    }
}

void hwops_set_all(int hw)
{
    hwops_mask = hw ? hw_caps() : 0;
}

/* Print */
/*-------*/

void hwops_print(void)
{
    uint32_t caps = hw_caps();
    puts("op            mode  (hw available)");
    for (int i = 0; i < OP_COUNT; i++) {
        if (i == OP_ENGINE_WAIT || i == OP_EMBED)
            continue;
        printf("%-12s %s   %s\n", names[i],
               ((hwops_mask >> i) & 1) ? "HW " : "CPU",
               ((caps >> i) & 1) ? "yes" : "no");
    }
}
