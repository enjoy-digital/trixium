/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium main_ram static area
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef TRIXIUM_STATIC_MEMORY_H
#define TRIXIUM_STATIC_MEMORY_H

// CPU main_ram static area, relative to TRIXIUM_HDR_ADDR: [0, 0x80000) BONS header/table,
// +0x80000 entry cache (model.c), +0xa0000 prompt buffer, +0xc0000 the
// buffers below; the heap starts at +0x100000.
// These buffers are only accessed after main_ram initialization and are NOT zeroed like .bss:
// every producer must initialize the bytes its consumer reads.
#define TRIXIUM_CONSOLE_LINE_ADDR   (TRIXIUM_HDR_ADDR + 0xc0000)
#define TRIXIUM_CONSOLE_LINE_SIZE   1024
#define TRIXIUM_CONSOLE_PROMPT_ADDR (TRIXIUM_HDR_ADDR + 0xc0400) // 64 int32s
#define TRIXIUM_FPROF_BUFFER_ADDR   (TRIXIUM_HDR_ADDR + 0xc0800) // 160 bytes, diagnostic only
#define TRIXIUM_RESIDENT_STATE_ADDR (TRIXIUM_HDR_ADDR + 0xc0900) // struct resident_state (resident.h)
#define TRIXIUM_HYBPROF_ADDR        (TRIXIUM_HDR_ADDR + 0xc0a00) // struct hyb_prof (hybrid.c), 256 B
#define TRIXIUM_EPS_MEMO_ADDR       (TRIXIUM_HDR_ADDR + 0xc0b00) // int32[128] seq_eps_int memo (infer_lists.c), 512 B
#define TRIXIUM_LISTPROF_ADDR       (TRIXIUM_HDR_ADDR + 0xc2000) // struct list_prof[16] (resident.c), 8.25 KB
#define TRIXIUM_RL_STATE_ADDR       (TRIXIUM_HDR_ADDR + 0xc6000) // struct rl_state (resident.c), < 1 KB
#define TRIXIUM_ECC_SAVED_ADDR      (TRIXIUM_HDR_ADDR + 0xc6400) // int8[1024] KV exponents for a re-run (infer.c), 1 KB
#define TRIXIUM_HYB_CKPT_ADDR       (TRIXIUM_HDR_ADDR + 0xc6800) // hybrid forward checkpoint (hybrid.c), < 1 KB
#define TRIXIUM_LIST_STAGE_ADDR     (TRIXIUM_HDR_ADDR + 0xc8000) // uint32[128][16] staged descriptor lists (resident.c), 8 KB

#endif
