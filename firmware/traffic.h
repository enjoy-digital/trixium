// SPDX-License-Identifier: BSD-2-Clause
#ifndef BONSAI_TRAFFIC_H
#define BONSAI_TRAFFIC_H

// Passive memory counters. Calls bracket perf_reset -> perf_print, NOT an
// individual op or decode-only interval. No MMIO when the generated map lacks it.
void traffic_begin(void);
void traffic_end(void);
void traffic_print(void);

#endif
