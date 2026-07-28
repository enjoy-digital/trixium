// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

#include <generated/csr.h>
#include <generated/soc.h>

#include "vec.h"
#include "perf.h"

#ifdef CSR_VECUNIT_BASE

#include "progs.h"

void vec_init(void)
{
    vecunit_prog_adr_write(0);
    for (unsigned i = 0; i < sizeof(bonsai_progs)/sizeof(bonsai_progs[0]); i++)
        vecunit_prog_write(bonsai_progs[i]);
}

uint64_t vec_run(uint16_t pc, const int32_t *args, int n_args)
{
    static void (* const wr[8])(uint32_t) = {
        vecunit_arg0_write, vecunit_arg1_write, vecunit_arg2_write, vecunit_arg3_write,
        vecunit_arg4_write, vecunit_arg5_write, vecunit_arg6_write, vecunit_arg7_write,
    };
    // Delta CSR writes: nothing else touches these registers, and each uncached CSR
    // write costs ~100 CPU cycles - per-head op sequences change only 1-2 args.
    static uint32_t cached[8];
    static uint32_t cached_pc = 0xFFFFFFFF;
    static uint8_t  valid;
    uint32_t done = vecunit_done_count_read();
    for (int i = 0; i < n_args; i++) {
        uint32_t v = (uint32_t)args[i];
        if (!(valid & (1u << i)) || cached[i] != v) {
            wr[i](v);
            cached[i] = v;
            valid |= (uint8_t)(1u << i);
        }
    }
    if (cached_pc != pc) {
        vecunit_pc_write(pc);
        cached_pc = pc;
    }
    vecunit_submit_write(1);
    while (vecunit_done_count_read() == done);
    return vecunit_res6_read();
}

#endif
