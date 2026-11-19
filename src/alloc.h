#pragma once

#include <assert.h>
#include <stdint.h>
#include <string.h>

// zig's FixedBufferAllocator: hands out zeroed memory from a fixed buffer, reset frees everything at once
typedef struct {
    unsigned char *buf;
    int64_t cap;
    int64_t end;
} FixedBufferAllocator;

// sizes round up to 8 bytes, so every allocation stays 8-byte aligned
static inline void *fba_alloc(FixedBufferAllocator *a, int64_t bytes) {
    bytes = (bytes + 7) / 8 * 8;
    assert(a->end + bytes <= a->cap && "out of memory, raise the buffer size in alloc.c");
    void *p = &a->buf[a->end];
    a->end += bytes;
    memset(p, 0, (size_t)bytes);
    return p;
}

static inline void fba_reset(FixedBufferAllocator *a) { a->end = 0; }

// all memory there is, sized at compile time. nothing calls malloc
extern FixedBufferAllocator perm_mem;  // params, their grads and optimizer state, never reset
extern FixedBufferAllocator graph_mem; // nodes, op outputs, their grads and scratch, reset after each backward
