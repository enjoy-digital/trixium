/* test/ddr shim: the CPU's system.h of a native liblitedram build (nothing to flush). */
#ifndef __SYSTEM_H
#define __SYSTEM_H

static inline void flush_cpu_icache(void) {}
static inline void flush_cpu_dcache(void) {}
static inline void flush_l2_cache(void) {}

#endif
