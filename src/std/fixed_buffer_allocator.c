#include "fixed_buffer_allocator.h"
#include <assert.h>
#include <stdint.h>

FixedBufferAllocator fixed_buffer_allocator_init(unsigned char *buffer, size_t buffer_len) { return (FixedBufferAllocator){.buffer = buffer, .buffer_len = buffer_len, .end_index = 0}; }

// compares addresses as integers like zig's @intFromPtr, pointers into different objects are not comparable in c
static bool owns_slice(const FixedBufferAllocator *self, const void *buf, size_t len) {
    uintptr_t start = (uintptr_t)self->buffer, p = (uintptr_t)buf;
    return p >= start && p - start <= self->buffer_len && len <= self->buffer_len - (p - start);
}

// misses an allocation whose successor was padded for alignment and freed, the padding stays used, as in zig
static bool is_last_allocation(const FixedBufferAllocator *self, const void *buf, size_t len) { return (uintptr_t)buf + len == (uintptr_t)(self->buffer + self->end_index); }

void *fixed_buffer_allocator_alloc(FixedBufferAllocator *self, size_t n, size_t alignment) {
    assert(alignment > 0 && (alignment & (alignment - 1)) == 0 && "alignment must be a power of two");
    uintptr_t end = (uintptr_t)(self->buffer + self->end_index);
    size_t adjusted_index = self->end_index + (alignment - end % alignment) % alignment;
    if (adjusted_index > self->buffer_len || n > self->buffer_len - adjusted_index) {
        return NULL;
    }
    self->end_index = adjusted_index + n;
    return self->buffer + adjusted_index;
}

bool fixed_buffer_allocator_resize(FixedBufferAllocator *self, void *buf, size_t len, size_t new_len) {
    assert(owns_slice(self, buf, len));
    if (!is_last_allocation(self, buf, len)) {
        return new_len <= len;
    }
    if (new_len <= len) {
        self->end_index -= len - new_len;
        return true;
    }
    if (new_len - len > self->buffer_len - self->end_index) {
        return false;
    }
    self->end_index += new_len - len;
    return true;
}

void fixed_buffer_allocator_free(FixedBufferAllocator *self, void *buf, size_t len) {
    assert(owns_slice(self, buf, len));
    if (is_last_allocation(self, buf, len)) {
        self->end_index -= len;
    }
}

void fixed_buffer_allocator_reset(FixedBufferAllocator *self) { self->end_index = 0; }
