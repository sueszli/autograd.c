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
// arena.alloc hands out the next zeroed block and moves used forward.
// arena.reset moves used back to 0, freeing every block at once.
// running out of space is a bug, so alloc asserts instead of returning NULL.
//
// example:
// --------
//     static u8 buf[1024];
//     Arena a = {.buf = buf, .cap = sizeof(buf)};
//     f64 *x = arena.alloc(&a, 8 * sizeof(f64)); // used = 64
//     arena.reset(&a);                           // used = 0
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

typedef struct {
    void *(*alloc)(Arena *a, usize bytes); // zeroed, aligned to max_align_t, asserts when full
    void (*reset)(Arena *a);               // frees every block at once
} ArenaNamespace;

extern const ArenaNamespace arena;
