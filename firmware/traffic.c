// SPDX-License-Identifier: BSD-2-Clause
#include <stdint.h>
#include <stdio.h>
#include <generated/csr.h>
#include <generated/soc.h>
#include "perf.h"
#include "traffic.h"
#include "static_memory.h"

#ifdef CSR_ZTRAFFIC_VERSION_ADDR
#define TRAFFIC_MAX_SOURCES 64
#define TRAFFIC_METRICS 5
struct traffic_source {
    uint32_t info, mask;
    uint64_t value[TRAFFIC_METRICS];
};
struct traffic_snapshot {
    uint64_t epoch, cycles;
    uint32_t count;
    struct traffic_source source[TRAFFIC_MAX_SOURCES];
};
// SRAM is only 8 KiB INCLUDING the stack. Use the established main_ram static
// window: +0x80000 entry cache, +0x90000 undo, +0xa0000 prompt, +0x100000 heap.
// Each capture initializes every field it reads; this window needs no BSS clear.
_Static_assert(2*sizeof(struct traffic_snapshot) <= BONSAI_TRAFFIC_BUFFER_SIZE,
               "traffic static window overflow");
#ifdef BONSAI_TRAFFIC_HOST_TEST
static struct traffic_snapshot buffers[2];
#else
static struct traffic_snapshot *const buffers =
    (struct traffic_snapshot *)BONSAI_TRAFFIC_BUFFER_ADDR;
#endif
#define start (buffers[0])
#define end   (buffers[1])
static const char *failure = "no_start";
static uint64_t window;
static int ended;

static const char *capture(struct traffic_snapshot *s)
{
    if (ztraffic_version_read() != 1) return "version";
    s->count = ztraffic_count_read();
    if (!s->count || s->count > TRAFFIC_MAX_SOURCES) return "source_count";
    ztraffic_snapshot_write(1);
    s->epoch = ztraffic_epoch_read();
    s->cycles = ztraffic_cycles_read();
    for (uint32_t i = 0; i < s->count; i++) {
        struct traffic_source *p = &s->source[i];
        ztraffic_select_write(i);
        if (!ztraffic_valid_read()) return "reset_or_selector";
        p->info = ztraffic_info_read();
        p->mask = ztraffic_mask_read();
        uint32_t kind = p->info & 255;
        if (!(p->info >> 16) || !((kind == 1 && p->mask == 31) ||
                                  (kind == 2 && p->mask == 7))) return "descriptor";
        for (uint32_t j = 0; j < i; j++)
            if ((s->source[j].info & 65535) == (p->info & 65535)) return "duplicate";
        p->value[0] = ztraffic_read_bytes_read();
        p->value[1] = ztraffic_write_bytes_read();
        p->value[2] = ztraffic_cmd_stalls_read();
        p->value[3] = ztraffic_data_stalls_read();
        p->value[4] = ztraffic_errors_read();
    }
    // Another producer must not use this shared snapshot bank during a capture.
    if (s->epoch != ztraffic_epoch_read() || s->cycles != ztraffic_cycles_read())
        return "snapshot_replaced";
    return 0;
}

void traffic_begin(void)
{
    window++;
    ended = 0;
    failure = capture(&start);
}

void traffic_end(void)
{
    if (ended) return;
    ended = 1;
    if (failure) return;
    failure = capture(&end);
    if (failure) return;
    if (start.epoch != end.epoch || end.cycles <= start.cycles || start.count != end.count) {
        failure = "reset_or_clock";
        return;
    }
    for (uint32_t i = 0; i < start.count; i++) {
        struct traffic_source *a = &start.source[i], *b = &end.source[i];
        if (a->info != b->info || a->mask != b->mask) {
            failure = "descriptor_changed";
            return;
        }
        for (int j = 0; j < TRAFFIC_METRICS; j++) {
            if (b->value[j] < a->value[j] || (!(a->mask & (1u << j)) && b->value[j])) {
                failure = "counter_reset_or_mask";
                return;
            }
        }
    }
}

static void field(const char *name, uint64_t value)
{
    printf(" %s=", name);
    perf_print_u64(value);
}

void traffic_print(void)
{
    static const char *names[TRAFFIC_METRICS] = {
        "read_bytes", "write_bytes", "cmd_stalls", "data_stalls", "errors"
    };
    if (!ended || failure) {
        printf("MEM_TRAFFIC v=1 valid=0 reason=%s\n", ended ? failure : "no_end");
        return;
    }
    for (uint32_t i = 0; i < start.count; i++) {
        struct traffic_source *a = &start.source[i], *b = &end.source[i];
        printf("MEM_TRAFFIC v=1 valid=1");
        field("window", window);
        field("sources", start.count);
        field("source", i);
        field("kind", a->info & 255);
        field("index", (a->info >> 8) & 255);
        field("word_bytes", a->info >> 16);
        field("mask", a->mask);
        field("epoch", start.epoch);
        field("cycles", end.cycles - start.cycles);
        field("clock_hz", CONFIG_CLOCK_FREQUENCY);
        for (int j = 0; j < TRAFFIC_METRICS; j++) field(names[j], b->value[j] - a->value[j]);
        putchar('\n');
    }
}
#else
void traffic_begin(void) {}
void traffic_end(void) {}
void traffic_print(void) { puts("MEM_TRAFFIC v=1 valid=0 reason=absent"); }
#endif
