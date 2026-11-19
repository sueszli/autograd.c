#pragma once

#include "types.h"

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
//     static u8 buf[1024];
//     Arena a = {.buf = buf, .cap = sizeof(buf)};
//     f64 *x = arena_alloc(&a, 8 * sizeof(f64)); // used = 64
//     arena_reset(&a);                                 // used = 0
//
// differences from zig:
// ---------------------
//     - alloc zeroes memory and asserts when full
//     - every block is aligned to max_align_t
//     - there is no free or resize, only reset
//
typedef struct {
    u8 *buf;
    usize cap;
    usize used;
} Arena;

void *arena_alloc(Arena *a, usize bytes);
void arena_reset(Arena *a);
