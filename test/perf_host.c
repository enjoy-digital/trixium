// SPDX-License-Identifier: BSD-2-Clause
// Native harness for the production profiler, with deterministic clock/CSR injection.
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "perf.h"
#include "bench.h"

uint64_t test_clock, test_live[4], test_snap[4];
uint32_t eng_poll_spins, eng_poll_waits, vec_poll_spins, vec_poll_waits;
uint32_t seq_poll_spins, seq_poll_waits;
void hybrid_stats_reset(void) {}
void hybrid_stats_print(void) {}
uint64_t perf_now64(void) { return test_clock; }
uint32_t perf_now(void) { return (uint32_t)test_clock; }

static void advance(uint64_t dt)
{
    test_clock += dt;
    test_live[0] += dt;
    test_live[1] += dt/2;
    test_live[2] += dt/4;
    test_live[3] += dt/8;
}

int main(void)
{
    struct bench_samples samples = {.start = 100};
    assert(bench_decode_wall(&samples) == 0);
    bench_token_ready(&samples, 150); // TTFT = 50, not a decode token.
    bench_token_emitted(&samples, 157);
    assert(bench_decode_compute(&samples) == 0);
    bench_token_ready(&samples, 217); // 60 compute + 7 UART cycles.
    bench_token_emitted(&samples, 222);
    bench_token_ready(&samples, 222 + (1ULL << 33));
    bench_token_emitted(&samples, 231 + (1ULL << 33));
    assert(samples.count == 3);
    assert(samples.first - samples.start == 50);
    assert(samples.emit_cycles == 21);
    assert(bench_decode_wall(&samples) == (1ULL << 33) + 72);
    assert(bench_decode_compute(&samples) == (1ULL << 33) + 60);

    perf_init();
    perf_enter(OP_EMBED);
    advance((1ULL << 32) + 17);
    perf_exit(OP_EMBED);
    assert(perf_cycles[OP_EMBED] == (1ULL << 32) + 17);
    perf_enter(OP_EMBED);
    advance(31);
    perf_exit(OP_EMBED);
    assert(perf_cycles[OP_EMBED] == (1ULL << 32) + 48);
    assert(perf_calls[OP_EMBED] == 2);
    perf_print();

    // A sum of short windows must not wrap, including on legacy low-word hardware.
    uint64_t acc[4] = {0};
    for (int i = 0; i < 4; i++) {
        perf_ovl_mark();
        advance(1ULL << 31);
        perf_ovl_acc(acc);
    }
#ifdef TEST_NO_COUNTERS
    assert(acc[0] == UINT64_MAX);
#else
    assert(acc[0] == (1ULL << 33));
    assert(acc[1] == (1ULL << 32));
    assert(acc[2] == (1ULL << 31));
    assert(acc[3] == (1ULL << 30));
#endif
    perf_ovl_report("short-sum", acc);

    uint64_t long_acc[4] = {0};
    perf_ovl_mark();
    advance((1ULL << 33) + 80);
    perf_ovl_acc(long_acc);
#ifdef TEST_WIDE_COUNTERS
    assert(long_acc[0] == (1ULL << 33) + 80);
    // A reset between snapshots must not be presented as huge utilization.
    perf_ovl_mark();
    for (int i = 0; i < 4; i++) test_live[i] = 0;
    advance(8);
    perf_ovl_acc(long_acc);
#endif
    assert(long_acc[0] == UINT64_MAX);
    perf_ovl_mark();
    advance(16);
    perf_ovl_acc(long_acc); // invalid is sticky until the caller clears the accumulator
    assert(long_acc[0] == UINT64_MAX);

    perf_reset();
    for (int i = 0; i < OP_COUNT; i++) {
        assert(perf_cycles[i] == 0);
        assert(perf_calls[i] == 0);
    }
    printf("format: "); perf_print_u64(0); putchar(' ');
    perf_print_u64(UINT64_MAX); putchar('\n');
    return 0;
}
