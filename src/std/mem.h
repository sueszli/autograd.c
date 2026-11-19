#pragma once

#include <stddef.h>

//
// a static arena over a buffer the caller owns.
// based on zig's FixedBufferAllocator:
// https://codeberg.org/ziglang/zig/src/branch/master/lib/std/heap/FixedBufferAllocator.zig
//
//     buf: [ used | new | ........ free ........ ]
//          0            ^used                   ^cap
//
// alloc hands out the next zeroed block and moves used forward.
// reset moves used back to 0, freeing every block at once.
// running out of space is a bug, so alloc asserts instead of returning NULL.
//
// example:
// --------
//     static unsigned char buf[1024];
//     Arena a = {.buf = buf, .cap = sizeof(buf)};
//     double *x = arena_alloc(&a, 8 * sizeof(double)); // used = 64
//     arena_reset(&a);                                 // used = 0
//
// differences from zig:
// ---------------------
//     - alloc zeroes memory and asserts when full
//     - every block is aligned to max_align_t
//     - there is no free or resize, only reset
//
typedef struct {
    unsigned char *buf;
    size_t cap;
    size_t used;
} Arena;

void *arena_alloc(Arena *a, size_t bytes);
void arena_reset(Arena *a);
