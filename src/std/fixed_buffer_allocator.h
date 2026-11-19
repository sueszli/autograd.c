#pragma once

#include <stdbool.h>
#include <stddef.h>

/*
 * port of zig's std.heap.FixedBufferAllocator: bumps end_index through a buffer the caller owns.
 * alloc returns NULL when full and leaves memory uninitialized. free and resize
 * only move end_index for the most recent allocation, reset frees everything at once.
 *
 * this ports the allocator's own functions, not the std.mem.Allocator interface in front
 * of them, so zero-length requests reach the allocator instead of being short-circuited.
 * a size that overflows returns NULL or false, where zig panics in safe builds.
 */
typedef struct {
    unsigned char *buffer;
    size_t buffer_len;
    size_t end_index;
} FixedBufferAllocator;

FixedBufferAllocator fixed_buffer_allocator_init(unsigned char *buffer, size_t buffer_len);
void *fixed_buffer_allocator_alloc(FixedBufferAllocator *self, size_t n, size_t alignment); // alignment is a power of two
bool fixed_buffer_allocator_resize(FixedBufferAllocator *self, void *buf, size_t len, size_t new_len);
void fixed_buffer_allocator_free(FixedBufferAllocator *self, void *buf, size_t len);
void fixed_buffer_allocator_reset(FixedBufferAllocator *self);
