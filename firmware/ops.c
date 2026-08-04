// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

#include <stdio.h>

#include <generated/csr.h>
#include <generated/soc.h>

#include "ops.h"

// Every op the gateware implements, on by default: each one is gated bit-exact against the
// interpreter (VECCRC) and token-gated in simulation, and the CPU paths are debug fallbacks.
// This used to be left at 0, so a card booted into the slowest configuration it had and the
// accelerators sat idle unless someone typed `hwops all hw` - measured on the 1.7B at
// 2.62e9 cycles / 6 forwards = 2.18 s per token with everything on the softcore.
//
// hwops_init() is called after the model header is known; `hwops all cpu` still gets the
// fallback path for A/B work.
uint32_t hwops_mask;

// All-CPU by default, which is slow but finishes. The HW attention path wedges the engine
// mid-op once a KV channel carries an extra row - position 9 with 8 channels, i.e. the fifth
// token of any real prompt - and a wedged engine holds the scratchpad grant until the FPGA is
// reconfigured (a CPU reset does not clear it, and reconfiguring wipes HBM). It is not the
// sequencer: turning the descriptor lists off (`seq off`, the default) moves the same stall
// onto the CPU-dispatched path, reported as ENGSTALL with the extra-row channel far behind
// the others on AR counts.
//
// The fast path is real and correct as far as it gets - `hwops all hw` measures 1.412 tok/s
// against 0.192 all-CPU (7.4x, two alternating runs each) and answers correctly - so it is
// worth having once the stall is fixed. Until then it is opt-in.
//
// Do NOT reach for a partial mask as a compromise: the resident-BFP loop keeps activations in
// the scratchpad, so a mixed mask pays a float round-trip every layer and lands at 0.121
// tok/s, slower than either end.
uint32_t hwops_default(void)
{
    return 0;
}

void hwops_init(void)
{
    hwops_mask = hwops_default();
}

static const char *names[OP_COUNT] = {
    "embed", "rmsnorm", "qknorm", "rope", "kvquant", "scores", "softmax", "wv",
    "quant8", "norm_i32", "silu_mul", "resid_add", "lmhead", "engine_wait",
};

const char *op_name(enum perf_op op)
{
    return (op < OP_COUNT) ? names[op] : "?";
}

uint32_t hw_caps(void)
{
    uint32_t caps = 0;
#ifdef BONSAI_HAS_VECUNIT
    caps |= (1 << OP_SILU_MUL) | (1 << OP_QUANT8) | (1 << OP_NORM_I32) | (1 << OP_RESID_ADD);
#endif
#ifdef BONSAI_HAS_VECUNIT2
    caps |= (1 << OP_RMSNORM);
#endif
#ifdef BONSAI_HAS_KVSTORE
    // KVQUANT/SCORES/WV all ride the KV regions + engine (AXPY is in the base engine).
    caps |= (1 << OP_KVQUANT) | (1 << OP_SCORES) | (1 << OP_WV);
#endif
#ifdef BONSAI_HAS_TOPK
    caps |= (1 << OP_LMHEAD);
#endif
#if defined(BONSAI_HAS_KVSTORE) && defined(BONSAI_HAS_VECUNIT2)
    // Fused resident attention: softmax8 + AXPY + reduce, and the resident front
    // (qk-norm/rope on the vector unit, image rope tables, resident kv-pack).
    caps |= (1 << OP_SOFTMAX) | (1 << OP_QKNORM) | (1 << OP_ROPE);
#endif
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
