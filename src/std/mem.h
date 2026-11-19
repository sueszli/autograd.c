#pragma once

#include "types.h"

//
// a static arena over a buffer the caller owns.
// based on zig's FixedBufferAllocator, without free or resize.
// https://codeberg.org/ziglang/zig/src/branch/master/lib/std/heap/FixedBufferAllocator.zig
//
//     buf: [ used | new | ........ free ........ ]
//          0            ^used                   ^cap
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
