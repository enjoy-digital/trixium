/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium token benchmark: ready-to-ready timing
 *
 * Kept independent of the clock and the console so the host tests can compile it natively.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef TRIXIUM_BENCH_H
#define TRIXIUM_BENCH_H

#include <stdint.h>

struct bench_samples {
    uint64_t start, first, last, emit_cycles, prior_emit_cycles;
    uint32_t count;
};

static inline void bench_token_ready(struct bench_samples *s, uint64_t now)
{
    if (!s->count)
        s->first = now;
    s->last = now;
    s->prior_emit_cycles = s->emit_cycles;
    s->count++;
}

static inline void bench_token_emitted(struct bench_samples *s, uint64_t now)
{
    s->emit_cycles += now - s->last;
}

static inline uint64_t bench_decode_wall(const struct bench_samples *s)
{
    return s->count > 1 ? s->last - s->first : 0;
}

static inline uint64_t bench_decode_compute(const struct bench_samples *s)
{
    uint64_t wall = bench_decode_wall(s);
    return wall >= s->prior_emit_cycles ? wall - s->prior_emit_cycles : 0;
}

#endif
