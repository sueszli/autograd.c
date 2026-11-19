#pragma once

#include <assert.h>
#include <stdint.h>
#include <string.h>

// fixed-size buffer with a top offset. push hands out zeroed memory, reset drops everything at once
typedef struct {
    unsigned char *buf;
    int64_t cap;
    int64_t len;
} Stack;

// sizes round up to 8 bytes, so pushes of one 8-byte-aligned type stay contiguous and indexable
static inline void *stack_push(Stack *s, int64_t bytes) {
    bytes = (bytes + 7) / 8 * 8;
    assert(s->len + bytes <= s->cap && "stack full, raise its capacity");
    void *p = &s->buf[s->len];
    s->len += bytes;
    memset(p, 0, (size_t)bytes);
    return p;
}

static inline void stack_reset(Stack *s) { s->len = 0; }
