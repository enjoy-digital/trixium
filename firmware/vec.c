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

uint32_t vec_poll_spins, vec_poll_waits;

uint64_t vec_run(uint16_t pc, const int32_t *args, int n_args)
{
    static void (* const wr[8])(uint32_t) = {
        vecunit_arg0_write, vecunit_arg1_write, vecunit_arg2_write, vecunit_arg3_write,
        vecunit_arg4_write, vecunit_arg5_write, vecunit_arg6_write, vecunit_arg7_write,
    };
    uint32_t done = vecunit_done_count_read();
    for (int i = 0; i < n_args; i++)
        wr[i]((uint32_t)args[i]);
    vecunit_pc_write(pc);
    vecunit_submit_write(1);
    // Spin-capped like engine_run/seq_run: a program that never finishes must say so.
    // This was the last silent-forever wait - a vecunit wedge showed up as a sim (or card)
    // that just stopped printing, with nothing to distinguish it from an engine stall.
    // The status CSR's bad_op field also names a corrupted/unknown instruction directly.
    uint32_t spins = 0, reports = 0;
    vec_poll_waits++;
    while (vecunit_done_count_read() == done) {
        vec_poll_spins++;
        if (++spins < 400000u)
            continue;
        spins = 0;
        printf("VECSTALL pc=%u status=%02lx\n",
               pc, (unsigned long)vecunit_status_read());
        if (++reports == 8) {
            puts("VECSTALL: giving up (results from here are garbage).");
            break;
        }
    }
    return vecunit_res6_read();
}

#endif
