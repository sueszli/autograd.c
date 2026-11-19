#pragma once

#include <stdbool.h>
#include <stddef.h>

//
// a static arena with one fixed buffer, one index into it and no malloc.
// based on zig's FixedBufferAllocator:
// https://codeberg.org/ziglang/zig/src/branch/master/lib/std/heap/FixedBufferAllocator.zig
//
//     buffer: [ used | pad | new | ........ free ........ ]
//             0                  ^end_index             ^buffer_len
//
// alloc moves end_index forward. reset moves it back to 0, freeing everything at once.
// free and resize only act on the most recent allocation.
// alloc returns NULL when full. it does not zero memory.
//
// example:
// --------
//     alignas(8) unsigned char buf[64];
//     FixedBufferAllocator fba = fixed_buffer_allocator_init(buf, sizeof(buf));
//     double *x = fixed_buffer_allocator_alloc(&fba, 8 * sizeof(double), alignof(double)); // end_index = 64
//     fixed_buffer_allocator_alloc(&fba, 1, 1);                                            // NULL, full
//     fixed_buffer_allocator_reset(&fba);                                                  // end_index = 0
//
// differences from zig:
// ---------------------
//     - no std.mem.Allocator interface, so zero-length requests reach the allocator
//     - overflow returns NULL or false. zig panics
//     - no remap and no thread-safe variant
//
typedef struct {
    unsigned char *buffer;
    size_t buffer_len;
    size_t end_index;
} FixedBufferAllocator;

FixedBufferAllocator fixed_buffer_allocator_init(unsigned char *buffer, size_t buffer_len);
void *fixed_buffer_allocator_alloc(FixedBufferAllocator *self, size_t n, size_t alignment); // alignment must be a power of 2
bool fixed_buffer_allocator_resize(FixedBufferAllocator *self, void *buf, size_t len, size_t new_len);
void fixed_buffer_allocator_free(FixedBufferAllocator *self, void *buf, size_t len);
void fixed_buffer_allocator_reset(FixedBufferAllocator *self);
