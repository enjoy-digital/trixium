// SPDX-License-Identifier: BSD-2-Clause
// Deterministic CSR injection for the actual firmware traffic capture/reporting.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <generated/csr.h>
#include "traffic.h"

uint32_t test_version = 1, test_count = 2, test_valid = 1, test_select;
uint64_t test_epoch = 7, test_cycles = (1ULL << 48), test_values[2][5];
uint32_t test_info[2] = {1 | (32 << 16), 2 | (64 << 16)};
uint32_t test_mask[2] = {31, 7};
int test_conflict;

void perf_print_u64(uint64_t v) { printf("%llu", (unsigned long long)v); }

int main(int argc, char **argv)
{
    int mode = argc > 1 ? atoi(argv[1]) : 0;
    for (int i = 0; i < 2; i++) {
        test_values[i][0] = (1ULL << 40);
        test_values[i][1] = (1ULL << 40);
    }
    if (mode == 3) test_count = 65;
    if (mode == 5) test_valid = 0;
    if (mode == 6) test_version = 2;
    if (mode == 9) { test_info[1] = test_info[0]; test_mask[1] = 31; }
    if (mode == 11) test_conflict = 1;
    if (mode != 14) traffic_begin();
    test_cycles += mode == 8 ? 0 : 500;
    for (int i = 0; i < 2; i++) {
        test_values[i][0] += 128;
        test_values[i][1] += 64;
        test_values[i][2] += 5;
    }
    if (mode == 1) test_epoch++;
    if (mode == 2) test_values[0][0] = 0;
    if (mode == 4) test_info[0] += 256;
    if (mode == 7) test_count = 1;
    if (mode == 10) test_values[1][4] = 1;
    if (mode != 12) traffic_end();
    if (mode == 13) { test_cycles += 100; traffic_end(); } // Idempotent end.
    traffic_print();
    return 0;
}
