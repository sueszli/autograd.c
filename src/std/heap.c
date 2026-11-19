#include "heap.h"
#include <assert.h>
#include <stdint.h>

FixedBufferAllocator fixed_buffer_allocator_init(unsigned char *buffer, size_t buffer_len) {
    // an empty arena over the caller's buffer
    return (FixedBufferAllocator){.buffer = buffer, .buffer_len = buffer_len, .end_index = 0};
}

static bool owns_slice(const FixedBufferAllocator *self, const void *buf, size_t len) {
    // integer addresses like zig's @intFromPtr, c can't compare pointers into different objects
    uintptr_t start = (uintptr_t)self->buffer, p = (uintptr_t)buf;
    return p >= start && p - start <= self->buffer_len && len <= self->buffer_len - (p - start);
}

static bool is_last_allocation(const FixedBufferAllocator *self, const void *buf, size_t len) {
    // misses a block whose successor was padded and then freed, the padding stays used. same as zig
    return (uintptr_t)buf + len == (uintptr_t)(self->buffer + self->end_index);
}

void *fixed_buffer_allocator_alloc(FixedBufferAllocator *self, size_t n, size_t alignment) {
    // pad end_index up to the alignment, then bump it past n bytes, or NULL if that overflows the buffer
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
    // the last block grows or shrinks in place. older blocks can only shrink, and nothing is reclaimed
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
    // only the last block gives its bytes back, older ones wait for reset
    assert(owns_slice(self, buf, len));
    if (is_last_allocation(self, buf, len)) {
        self->end_index -= len;
    }
}

void fixed_buffer_allocator_reset(FixedBufferAllocator *self) {
    // frees every block at once
    self->end_index = 0;
}
