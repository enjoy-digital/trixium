// SPDX-License-Identifier: BSD-2-Clause
// The firmware's separate int->float cast, ldexpf scale, and FP32 multiply.
// Linux syscalls only, so both RV32 ilp32f and soft-float ilp32 link target libm.
#include <math.h>
#include <stdint.h>

// ldexpf's range-error branch references newlib errno. No libc startup is
// needed in this single-threaded numerical fixture; preserve its writable slot.
static int target_errno;
int *__errno(void) { return &target_errno; }

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
#ifdef __riscv_flen
    // Assert, do not silently change the reference's normal dynamic mode.
    unsigned frm;
    __asm__ volatile("frrm %0" : "=r"(frm));
    if (frm != 0) quit(4);
#endif
    int32_t input[2];
    while (transfer(0, input, sizeof(input))) {
        volatile float converted = (float)input[0];
        volatile float scale = ldexpf(1.0f, input[1]);
        float restored = converted * scale;
        transfer(1, &restored, sizeof(restored));
    }
    quit(0);
}
