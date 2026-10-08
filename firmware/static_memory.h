// This file is part of TriXium.
//
// Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
// SPDX-License-Identifier: BSD-2-Clause

#ifndef TRIXIUM_STATIC_MEMORY_H
#define TRIXIUM_STATIC_MEMORY_H

// Legacy CPU main_ram static area, relative to TRIXIUM_HDR_ADDR. Existing owners:
// [0,0x80000) header/table; +0x80000 entry cache; +0x90000 KV undo;
// +0xa0000 prompt buffer. The allocator starts at +0x100000.
// These buffers are accessed only after main_ram initialization. They are NOT
// BSS-initialized: every producer must initialize the bytes its consumer reads.
#define TRIXIUM_CONSOLE_LINE_ADDR   (TRIXIUM_HDR_ADDR + 0xc0000)
#define TRIXIUM_CONSOLE_LINE_SIZE   1024
#define TRIXIUM_CONSOLE_PROMPT_ADDR (TRIXIUM_HDR_ADDR + 0xc0400) // 64 int32s
#define TRIXIUM_FPROF_BUFFER_ADDR   (TRIXIUM_HDR_ADDR + 0xc0800) // 160 bytes, diagnostic only
#define TRIXIUM_RESIDENT_STATE_ADDR (TRIXIUM_HDR_ADDR + 0xc0900) // struct resident_state (resident.h)
#define TRIXIUM_HYBPROF_ADDR        (TRIXIUM_HDR_ADDR + 0xc0a00) // struct hyb_prof (hybrid.c), 256 B
#define TRIXIUM_EPS_MEMO_ADDR       (TRIXIUM_HDR_ADDR + 0xc0b00) // int32[128] seq_eps_int memo (infer.c), 512 B
#define TRIXIUM_LISTPROF_ADDR       (TRIXIUM_HDR_ADDR + 0xc2000) // struct list_prof[16] (resident.c), 8.6 KB

#endif
