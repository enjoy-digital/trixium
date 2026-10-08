// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

#include <generated/csr.h>
#include <generated/soc.h>

#include "vec.h"
#include "engine.h"
#include "perf.h"


#include "progs.h"

// The microcode encoding must be the bitstream's (model.asm docstring). The Makefile
// regenerates progs.h per build tree; this catches a stale header before it costs a boot.
#ifndef TRIXIUM_VEC_VREGS
#define TRIXIUM_VEC_VREGS 8
#endif
#if TRIXIUM_PROGS_VREGS != TRIXIUM_VEC_VREGS
#error "progs.h was emitted for a different vector register count than this bitstream"
#endif

void vec_init(void)
{
    trixium_vecunit_prog_adr_write(0);
    for (unsigned i = 0; i < sizeof(trixium_progs)/sizeof(trixium_progs[0]); i++)
        trixium_vecunit_prog_write(trixium_progs[i]);
}

uint32_t vec_poll_spins, vec_poll_waits;

uint64_t vec_run(uint16_t pc, const int32_t *args, int n_args)
{
    static void (* const wr[8])(uint32_t) = {
        trixium_vecunit_arg0_write, trixium_vecunit_arg1_write, trixium_vecunit_arg2_write, trixium_vecunit_arg3_write,
        trixium_vecunit_arg4_write, trixium_vecunit_arg5_write, trixium_vecunit_arg6_write, trixium_vecunit_arg7_write,
    };
    uint32_t done = trixium_vecunit_done_count_read();
    for (int i = 0; i < n_args; i++)
        wr[i]((uint32_t)args[i]);
    trixium_vecunit_pc_write(pc);
    trixium_vecunit_submit_write(1);
    // Spin-capped like engine_run/seq_run: a program that never finishes must say so.
    // This was the last silent-forever wait - a vecunit wedge showed up as a sim (or card)
    // that just stopped printing, with nothing to distinguish it from an engine stall.
    // The status CSR's bad_op field also names a corrupted/unknown instruction directly.
    uint32_t spins = 0, reports = 0;
    vec_poll_waits++;
    while (trixium_vecunit_done_count_read() == done) {
        vec_poll_spins++;
        if (++spins < 400000u)
            continue;
        spins = 0;
        printf("VECSTALL pc=%u status=%02lx\n",
               pc, (unsigned long)trixium_vecunit_status_read());
        if (++reports == 8) {
            puts("VECSTALL: giving up (results from here are garbage).");
            engine_wedged = 1;   // one flag: the generation aborts and engine_recover runs
            break;
        }
    }
    return trixium_vecunit_res6_read();
}

