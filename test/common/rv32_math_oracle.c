/* SPDX-License-Identifier: BSD-2-Clause
 *
 * Standalone Linux-syscall harness for qemu-riscv32. Link the SAME RV32F libm
 * archive as the firmware; do not substitute the workstation's expf.
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include <stdint.h>
#include "fastmath.h"

extern float __ieee754_expf(float);
static float lut[TRIXIUM_SILU_LUT_N + 1], slut[TRIXIUM_SILU_LUT_N + 1];

static long sys_call(long nr, long fd, void *buf, long size)
{
    register long a0 __asm__("a0") = fd;
    register void *a1 __asm__("a1") = buf;
    register long a2 __asm__("a2") = size;
    register long a7 __asm__("a7") = nr;
    __asm__ volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
    return a0;
}

static void quit(int code)
{
    sys_call(93, code, 0, 0);
    __builtin_unreachable();
}

static int transfer(int write, void *buf, long size)
{
    char *p = buf;
    long done = 0;
    while (done < size) {
        long n = sys_call(write ? 64 : 63, write ? 1 : 0, p + done, size - done);
        if (n < 0) quit(2);
        if (n == 0) {
            if (write || done) quit(3);
            return 0;
        }
        done += n;
    }
    return 1;
}

void _start(void)
{
    for (int i = 0; i <= TRIXIUM_SILU_LUT_N; i++) {
        float x = -TRIXIUM_SILU_LUT_LIMIT + (float)i*(1.0f/TRIXIUM_SILU_LUT_SCALE);
        lut[i] = x/(1.0f + __ieee754_expf(-x));
        slut[i] = trixium_sigmoid_point(x);
    }
    float x;
    while (transfer(0, &x, sizeof(x))) {
        float neg = x < 0.0f ? x : -x;     /* softmax arguments are s - max <= 0 */
        float out[5] = {__ieee754_expf(x), trixium_siluf(x), trixium_silu_lookup(x, lut),
                        trixium_sigmoid_lookup(x, slut), trixium_softmax_exp(neg)};
        transfer(1, out, sizeof(out));
    }
    quit(0);
}
