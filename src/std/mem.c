#include "mem.h"
#include <assert.h>
#include <string.h>

void *arena_alloc(Arena *a, usize bytes) {
    // pad the next address up to max_align_t, then bump used past the block
    const usize align = alignof(max_align_t);
    usize start = a->used + (align - (usize)(a->buf + a->used) % align) % align;
    assert(start <= a->cap && bytes <= a->cap - start && "arena full, raise its buffer size");
    a->used = start + bytes;
    return memset(a->buf + start, 0, bytes);
}

void arena_reset(Arena *a) {
    // frees every block at once
    a->used = 0;
}
