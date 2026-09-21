INCLUDE generated/output_format.ld
ENTRY(_start)

__DYNAMIC = 0;

INCLUDE generated/regions.ld

SECTIONS
{
	.text :
	{
		_ftext = .;
		/* Make sure crt0 files come first, and they, and the isr */
		/* don't get disposed of by greedy optimisation */
		*crt0*(.text)
		KEEP(*crt0*(.text))
		KEEP(*(.text.isr))

		*(.text .stub .text.* .gnu.linkonce.t.*)
		_etext = .;
	} > main_ram

	.rodata :
	{
		. = ALIGN(8);
		_frodata = .;
		*(.rodata .rodata.* .gnu.linkonce.r.*)
		*(.rodata1)
		*(.srodata .srodata.*)
		*(.got .got.*)
		*(.toc .toc.*)

		/* Make sure the file is aligned on disk as well
		   as in memory. */
		FILL(0);
		. = ALIGN(8);
		_erodata = .;
	} > main_ram

	.preinit_array :
	{
		. = ALIGN(8);
		PROVIDE_HIDDEN (__preinit_array_start = .);
		KEEP(*(SORT(.preinit_array.*)))
		KEEP(*(.preinit_array))
		PROVIDE_HIDDEN (__preinit_array_end = .);
	} > main_ram

	.init_array :
	{
		. = ALIGN(8);
		PROVIDE_HIDDEN (__init_array_start = .);
		KEEP(*(SORT(.init_array.*)))
		KEEP(*(.init_array))
		KEEP(*(SORT(.ctors.*)))
		KEEP(*(.ctors))
		PROVIDE_HIDDEN (__init_array_end = .);
	} > main_ram

	.fini_array :
	{
		. = ALIGN(8);
		PROVIDE_HIDDEN (__fini_array_start = .);
		KEEP(*(SORT(.fini_array.*)))
		KEEP(*(.fini_array))
		KEEP(*(SORT(.dtors.*)))
		KEEP(*(.dtors))
		PROVIDE_HIDDEN (__fini_array_end = .);
	} > main_ram

	.data :
	{
		. = ALIGN(8);
		_fdata = .;
		*(.data .data.* .gnu.linkonce.d.*)
		*(.data1)
		_gp = ALIGN(16);
		*(.sdata .sdata.* .gnu.linkonce.s.*)

		/* Make sure the file is aligned on disk as well
		   as in memory. */
		FILL(0);
		. = ALIGN(8);
		_edata = .;
	} > sram AT > main_ram

	.bss :
	{
		. = ALIGN(8);
		_fbss = .;
		*(.dynsbss)
		*(.sbss .sbss.* .gnu.linkonce.sb.*)
		*(.scommon)
		*(.dynbss)
		*(.bss .bss.* .gnu.linkonce.b.*)
		*(COMMON)
		. = ALIGN(8);
		_ebss = .;
		_end = .;
	} > sram
}

/* The stack lives at the top of sram and takes whatever .data/.bss leave it. That number
 * MUST stay >= ~4 KB: at 1 KB deep call paths overflowed into .bss and the firmware died by
 * silent corruption at a location that moved with every relink. The fix is keeping large
 * arrays OUT of .bss (they live in the main_ram static window instead - grep ECACHE_BASE),
 * NOT moving the stack to main_ram: on the card main_ram is the HBM channel-0 window, and
 * the one firmware built with its stack there coincided with back-to-back host crashes.
 * Check after adding statics:  riscv64-unknown-elf-nm bonsai.elf | grep -E "_ebss|_fstack" */
PROVIDE(_fstack = ORIGIN(sram) + LENGTH(sram));
ASSERT(_fstack - _ebss >= 4096, "firmware must leave at least 4 KiB of SRAM for stack")

PROVIDE(_fdata_rom = LOADADDR(.data));
PROVIDE(_edata_rom = LOADADDR(.data) + SIZEOF(.data));
