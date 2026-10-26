/* test/ddr shim: the generated soc.h of a native liblitedram build on the eye model (model.c).
 * The training's constants (SDRAM_WRITE_TRAINING, ...) come from the compiler command line, as
 * a target's add_constant() would put them here. */
#ifndef __GENERATED_SOC_H
#define __GENERATED_SOC_H
#include <stdint.h>

#define CONFIG_CLOCK_FREQUENCY 125000000
#define CONFIG_CPU_NOP "nop"
#define CONFIG_CSR_DATA_WIDTH 32
#define CONFIG_CSR_ALIGNMENT 32
#define CONFIG_CSR_ORDERING_BIG
#define CONFIG_BIOS_NO_DELAYS

/* The CSR bus is the eye model's: hw/common.h keeps its buffer accessors on top of these. */
#define CSR_ACCESSORS_DEFINED
unsigned long csr_read_simple(unsigned long a);
void csr_write_simple(unsigned long v, unsigned long a);

#endif
