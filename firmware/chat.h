/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium chat turns: the conversation the console `chat` and the web page share
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#ifndef TRIXIUM_CHAT_H
#define TRIXIUM_CHAT_H

#include <stdint.h>

struct chat_result {
    int      tokens;     // Reply tokens (the closing <|im_end|> included).
    uint32_t ms;         // Generation time (prefill + decode).
    int      context;    // Conversation positions after the turn.
    int      capped;     // The reply hit nmax before <|im_end|>.
    int      restarted;  // The context was full (no KV ring): the turn started a new conversation.
    int      error;      // Generation error: the conversation was cleared.
};

int  chat_ready(void);   // A model and a tokenizer are loaded.
void chat_reset(void);
// One turn: `line` in, the reply's UTF-8 bytes to `out` as they come (nmax tokens at most).
int  chat_turn(const char *line, int nmax, void (*out)(const uint8_t *b, uint32_t n),
               struct chat_result *res);

#endif
