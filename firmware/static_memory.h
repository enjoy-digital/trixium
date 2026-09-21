// SPDX-License-Identifier: BSD-2-Clause
#ifndef BONSAI_STATIC_MEMORY_H
#define BONSAI_STATIC_MEMORY_H

// Legacy CPU main_ram static area, relative to BONSAI_HDR_ADDR. Existing owners:
// [0,0x80000) header/table; +0x80000 entry cache; +0x90000 KV undo;
// +0xa0000 prompt buffer. The allocator starts at +0x100000.
// These buffers are accessed only after main_ram initialization. They are NOT
// BSS-initialized: every producer must initialize the bytes its consumer reads.
#define BONSAI_TRAFFIC_BUFFER_ADDR (BONSAI_HDR_ADDR + 0xb0000)
#define BONSAI_TRAFFIC_BUFFER_SIZE 0x10000
#define BONSAI_CONSOLE_LINE_ADDR   (BONSAI_HDR_ADDR + 0xc0000)
#define BONSAI_CONSOLE_LINE_SIZE   1024
#define BONSAI_CONSOLE_PROMPT_ADDR (BONSAI_HDR_ADDR + 0xc0400) // 64 int32s
#define BONSAI_ATRACE_BUFFER_ADDR  (BONSAI_HDR_ADDR + 0xc0500) // 192 uint32s
#define BONSAI_FPROF_BUFFER_ADDR   (BONSAI_HDR_ADDR + 0xc0800) // 160 bytes, diagnostic only

#endif
