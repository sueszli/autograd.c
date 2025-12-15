#include "utils/types.h"
#include <assert.h>
#include <stdlib.h>

#define CACHELINE_SIZE 64
_Static_assert(CACHELINE_SIZE >= sizeof(float32_t), "cacheline_size must be at least 4 bytes");
_Static_assert((CACHELINE_SIZE & (CACHELINE_SIZE - 1)) == 0, "cacheline_size must be power of 2");

static inline void *safe_aligned_alloc(uint64_t size_bytes) {
    size_t s_bytes = (size_t)size_bytes;
    if (s_bytes % CACHELINE_SIZE != 0) {
        s_bytes = (s_bytes / CACHELINE_SIZE + 1) * CACHELINE_SIZE;
    }
    void *ptr = aligned_alloc(CACHELINE_SIZE, s_bytes);
    assert(ptr != NULL && "aligned_alloc failed: out of memory");
    assert((uintptr_t)ptr % CACHELINE_SIZE == 0 && "allocated pointer is not properly aligned");
    return ptr;
}
