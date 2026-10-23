/* SPDX-License-Identifier: BSD-2-Clause
 *
 * TriXium byte-level BPE tokenizer (Qwen3 / Qwen3.5 BTOK tables)
 *
 * This file is part of TriXium.
 *
 * Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
 *
 */

#include <string.h>

#include "trixium_tok.h"

/* BTOK Tables */
/*-------------*/

/* BTOK (model/tokenizer.py): "BTOK", u32 version, u32 nvocab, u32 nmerges, [v2: 32-byte SHA-256],
 * nvocab x (u32 len, len bytes), nmerges x (u32 a, u32 b, u32 a+b). Little-endian. */

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

int ttok_same(const uint8_t *a, const uint8_t *b, size_t n)
{
    while (n && *a == *b) { a++; b++; n--; }
    return n == 0;
}

static uint64_t mkey(uint32_t a, uint32_t b) { return ((uint64_t)a << 32) | b; }

static uint64_t hash64(uint64_t x)
{
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    return x ^ (x >> 33);
}

/* The header's size and counts; 0 if not a BTOK file. */
static size_t btok_header(const uint8_t *b, size_t len, uint32_t *nvocab, uint32_t *nmerges)
{
    if (len < 16 || !ttok_same(b, (const uint8_t *)"BTOK", 4))
        return 0;
    uint32_t ver = rd32(b + 4);
    *nvocab  = rd32(b + 8);
    *nmerges = rd32(b + 12);
    size_t hdr = ver >= 2 ? 48 : 16;
    return len < hdr ? 0 : hdr;
}

static uint32_t merge_cap(uint32_t nmerges)
{
    uint32_t cap = 1;
    while (cap < 2*nmerges) cap <<= 1;
    return cap;
}

size_t ttok_arena_size(const uint8_t *btok, size_t len)
{
    uint32_t nv, nm;
    if (!btok_header(btok, len, &nv, &nm))
        return 0;
    return (size_t)merge_cap(nm)*sizeof(struct ttok_merge) + (size_t)nv*4u +
           TTOK_CHUNK_MAX*sizeof(int32_t);
}

int ttok_init(struct ttok *t, const uint8_t *btok, size_t len, void *arena)
{
    uint32_t nv, nm;
    size_t hdr = btok_header(btok, len, &nv, &nm);
    if (!hdr)
        return -1;
    memset(t, 0, sizeof(*t));
    t->sha_ok = rd32(btok + 4) >= 2;
    if (t->sha_ok)
        memcpy(t->sha, btok + 16, 32);
    uint32_t cap = merge_cap(nm);
    t->mtab    = (struct ttok_merge *)arena;     /* 8-byte members first: the arena's alignment. */
    t->voff    = (uint32_t *)(t->mtab + cap);
    t->scratch = (int32_t *)(t->voff + nv);
    t->mmask   = cap - 1;
    t->blob    = btok;
    t->nvocab  = nv;
    t->nmerges = nm;

    /* The vocabulary: one offset per record, the single-byte tokens. */
    for (int i = 0; i < 256; i++) t->byte_id[i] = -1;
    size_t p = hdr;
    for (uint32_t i = 0; i < nv; i++) {
        if (p + 4 > len) return -1;
        uint32_t n = rd32(btok + p);
        if (p + 4 + n > len) return -1;
        t->voff[i] = (uint32_t)p;
        if (n == 1 && t->byte_id[btok[p + 4]] < 0)
            t->byte_id[btok[p + 4]] = (int32_t)i;
        p += 4 + n;
    }
    /* The merges, ranked by their order. */
    if (p + (size_t)nm*12 > len) return -1;
    for (uint32_t i = 0; i < cap; i++) t->mtab[i].key = UINT64_MAX;
    for (uint32_t r = 0; r < nm; r++, p += 12) {
        uint64_t k = mkey(rd32(btok + p), rd32(btok + p + 4));
        uint32_t i = (uint32_t)(hash64(k) & t->mmask);
        while (t->mtab[i].key != UINT64_MAX) i = (i + 1) & t->mmask;
        t->mtab[i].key = k; t->mtab[i].rank = r; t->mtab[i].result = rd32(btok + p + 8);
    }
    t->im_start  = ttok_find(t, "<|im_start|>");
    t->im_end    = ttok_find(t, "<|im_end|>");
    t->think     = ttok_find(t, "<think>");
    t->think_end = ttok_find(t, "</think>");
    return 0;
}

const uint8_t *ttok_piece(const struct ttok *t, int id, uint32_t *len)
{
    if (id < 0 || (uint32_t)id >= t->nvocab) { *len = 0; return NULL; }
    *len = rd32(t->blob + t->voff[id]);
    return t->blob + t->voff[id] + 4;
}

int ttok_find(const struct ttok *t, const char *lit)
{
    uint32_t n = (uint32_t)strlen(lit), len;
    for (uint32_t i = t->nvocab; i-- > 0; ) {
        const uint8_t *b = ttok_piece(t, (int)i, &len);
        if (len == n && ttok_same(b, (const uint8_t *)lit, n))
            return (int)i;
    }
    return -1;
}

/* BPE */
/*-----*/

static const struct ttok_merge *merge_find(const struct ttok *t, uint32_t a, uint32_t b)
{
    uint64_t k = mkey(a, b);
    uint32_t i = (uint32_t)(hash64(k) & t->mmask);
    while (t->mtab[i].key != UINT64_MAX) {
        if (t->mtab[i].key == k)
            return &t->mtab[i];
        i = (i + 1) & t->mmask;
    }
    return NULL;
}

/* One pre-token: from single bytes, repeatedly merge the best-ranked adjacent pair. */
static int bpe_chunk(const struct ttok *t, const uint8_t *s, size_t n, int *out, int max)
{
    int32_t *ids = t->scratch;
    size_t cnt = 0;
    for (size_t i = 0; i < n; i++)
        if (t->byte_id[s[i]] >= 0)           /* A byte-level vocabulary covers 0..255. */
            ids[cnt++] = t->byte_id[s[i]];
    while (cnt > 1) {
        uint32_t best = UINT32_MAX, res = 0;
        size_t at = 0;
        for (size_t i = 0; i + 1 < cnt; i++) {
            const struct ttok_merge *m = merge_find(t, (uint32_t)ids[i], (uint32_t)ids[i+1]);
            if (m && m->rank < best) { best = m->rank; at = i; res = m->result; }
        }
        if (best == UINT32_MAX) break;
        ids[at] = (int32_t)res;
        memmove(&ids[at+1], &ids[at+2], (cnt - at - 2)*sizeof(ids[0]));
        cnt--;
    }
    int w = 0;
    for (size_t i = 0; i < cnt && w < max; i++) out[w++] = ids[i];
    return w;
}

/* Pre-tokenizer */
/*---------------*/

/* Minimal UTF-8 decode: the codepoint, *len its bytes (1 for an invalid lead byte). */
static uint32_t utf8_cp(const uint8_t *s, size_t n, int *len)
{
    if (n == 0) { *len = 0; return 0; }
    if (s[0] < 0x80)                     { *len = 1; return s[0]; }
    if ((s[0] & 0xE0) == 0xC0 && n >= 2) { *len = 2; return ((s[0]&0x1F)<<6)|(s[1]&0x3F); }
    if ((s[0] & 0xF0) == 0xE0 && n >= 3) { *len = 3;
        return ((s[0]&0x0F)<<12)|((s[1]&0x3F)<<6)|(s[2]&0x3F); }
    if ((s[0] & 0xF8) == 0xF0 && n >= 4) { *len = 4;
        return ((uint32_t)(s[0]&0x07)<<18)|((s[1]&0x3F)<<12)|((s[2]&0x3F)<<6)|(s[3]&0x3F); }
    *len = 1; return s[0];
}

/* \p{L} approximation: separates letters (accented, Greek, Cyrillic, CJK) from symbols and
 * emoji, which is what the split depends on. */
static int cp_is_letter(uint32_t cp)
{
    if (cp < 0x80)    return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
    if (cp < 0x2000)  return 1;   /* Latin-1/Extended, Greek, Cyrillic, Hebrew, Arabic... */
    if (cp < 0x3040)  return 0;   /* Punctuation, arrows, math, misc symbols.             */
    if (cp < 0xD800)  return 1;   /* Kana, Hangul, CJK.                                   */
    if (cp < 0x1F000) return 1;   /* CJK compat, presentation forms.                      */
    return 0;                     /* Emoji and pictographs.                               */
}
static int cp_is_digit(uint32_t cp) { return cp >= '0' && cp <= '9'; }
static int cp_is_space(uint32_t cp)
{
    return cp == ' ' || cp == '\t' || cp == '\r' || cp == '\n' || cp == 0x0b || cp == 0x0c;
}

/* (?i:'s|'t|'re|'ve|'m|'ll|'d): the contraction's length at s, or 0. */
static size_t contraction(const uint8_t *s, size_t n)
{
    static const char *const contr[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    if (n < 2 || s[0] != '\'')
        return 0;
    for (unsigned c = 0; c < sizeof(contr)/sizeof(contr[0]); c++) {
        size_t cl = strlen(contr[c]), k;
        if (1 + cl > n) continue;
        for (k = 0; k < cl && (s[1 + k] | 0x20) == (uint8_t)contr[c][k]; k++);
        if (k == cl) return 1 + cl;
    }
    return 0;
}

/* The Qwen/GPT-2 pre-tokenizer alternation, in order:
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}
 *   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
 * Order matters: a symbol run absorbs one leading space, so " + " is [" +"][" "]. */
int ttok_encode(const struct ttok *t, const uint8_t *s, size_t n, int *out, int max)
{
    size_t i = 0;
    int w = 0;
    while (i < n && w < max) {
        size_t start = i, cl;
        int l, kl;
        uint32_t cp = utf8_cp(s + i, n - i, &l);
        int done = 0;
        /* 1. Contractions. */
        if ((cl = contraction(s + i, n - i))) { i += cl; done = 1; }
        /* 2. [^\r\n\p{L}\p{N}]? \p{L}+ */
        if (!done) {
            size_t k = i; uint32_t kc = cp;
            if (!cp_is_letter(kc) && !cp_is_digit(kc) && kc != '\r' && kc != '\n') {
                k += l;
                kc = utf8_cp(s + k, n - k, &kl);
            }
            if (k < n && cp_is_letter(kc)) {
                while (k < n && cp_is_letter(utf8_cp(s + k, n - k, &kl))) k += kl;
                i = k; done = 1;
            }
        }
        /* 3. \p{N}: one digit at a time. */
        if (!done && cp_is_digit(cp)) { i += l; done = 1; }
        /* 4. ' ?' [^\s\p{L}\p{N}]+ [\r\n]*: a symbol run, absorbing one leading space. */
        if (!done) {
            size_t k = i; uint32_t kc = cp;
            if (kc == ' ') { k += l; kc = utf8_cp(s + k, n - k, &kl); }
            if (k < n && !cp_is_space(kc) && !cp_is_letter(kc) && !cp_is_digit(kc)) {
                while (k < n) {
                    uint32_t c2 = utf8_cp(s + k, n - k, &kl);
                    if (cp_is_space(c2) || cp_is_letter(c2) || cp_is_digit(c2)) break;
                    k += kl;
                }
                while (k < n && (s[k] == '\r' || s[k] == '\n')) k++;
                i = k; done = 1;
            }
        }
        /* 5/6/7. Whitespace: \s*[\r\n]+, then \s+(?!\S), then \s+. */
        if (!done) {
            size_t k = i;
            while (k < n && cp_is_space(utf8_cp(s + k, n - k, &kl))) k += kl;
            size_t nl_end = k;
            while (nl_end > i && (s[nl_end-1] == '\r' || s[nl_end-1] == '\n')) nl_end--;
            if (k > i && nl_end < k)
                i = k;                       /* The run ends in newline(s). */
            else if (k < n && k > i + 1)
                i = k - 1;                   /* \s+(?!\S): the last space goes with the word. */
            else
                i = (k > i) ? k : i + 1;
        }
        if (i <= start) i = start + 1;       /* Never stall. */
        /* A pre-token longer than the scratch (a huge unbroken run) in windows: never dropped. */
        for (size_t p = start; p < i && w < max; p += TTOK_CHUNK_MAX) {
            size_t m = i - p < TTOK_CHUNK_MAX ? i - p : TTOK_CHUNK_MAX;
            w += bpe_chunk(t, s + p, m, out + w, max - w);
        }
    }
    return w;
}

/* Chat Template */
/*---------------*/

static int lit(const struct ttok *t, const char *s, int *out, int max)
{
    return max > 0 ? ttok_encode(t, (const uint8_t *)s, strlen(s), out, max) : 0;
}

static int put(int id, int *out, int max)
{
    if (max <= 0) return 0;
    out[0] = id;
    return 1;
}

int ttok_turn(const struct ttok *t, const char *text, const char *system, int first,
              int prev_closed, int *out, int max)
{
    int n = 0;
    if (t->im_start < 0 || t->im_end < 0)
        return lit(t, text, out, max);
    if (!first) {
        if (!prev_closed) n += put(t->im_end, out + n, max - n);
        n += lit(t, "\n", out + n, max - n);
    }
    if (first && system && *system) {
        n += put(t->im_start, out + n, max - n);
        n += lit(t, "system\n", out + n, max - n);
        n += lit(t, system, out + n, max - n);
        n += put(t->im_end, out + n, max - n);
        n += lit(t, "\n", out + n, max - n);
    }
    n += put(t->im_start, out + n, max - n);
    n += lit(t, "user\n", out + n, max - n);
    n += lit(t, text, out + n, max - n);
    n += put(t->im_end, out + n, max - n);
    n += lit(t, "\n", out + n, max - n);
    n += put(t->im_start, out + n, max - n);
    n += lit(t, "assistant\n", out + n, max - n);
    if (t->think >= 0 && t->think_end >= 0) {
        n += put(t->think, out + n, max - n);
        n += lit(t, "\n\n", out + n, max - n);
        n += put(t->think_end, out + n, max - n);
        n += lit(t, "\n\n", out + n, max - n);
    }
    return n;
}
