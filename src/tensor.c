#include "tensor.h"
#include "std/mem.h"
#include <assert.h>
#include <string.h>

static alignas(16) u8 perm_buf[512 * 1024]; // parity uses about 200 KB: two param sets, their grads and AdamW state
static Arena perm = {.buf = perm_buf, .cap = sizeof(perm_buf)};

void *perm_alloc(i64 bytes) { return arena_alloc(&perm, (usize)bytes); }
void perm_reset(void) { arena_reset(&perm); }

void tensor_set_shape(Tensor *t, const i64 *shape, i64 ndim) {
    // copies the shape and multiplies it out into size
    assert(ndim >= 0 && ndim <= MAX_NDIM && "ndim out of range");
    t->ndim = ndim;
    t->size = 1;
    for (i64 d = 0; d < ndim; d++) {
        assert(shape[d] > 0 && "tensor dims must be positive");
        t->shape[d] = shape[d];
        t->size *= shape[d];
    }
}

Tensor *tensor_new(const i64 *shape, i64 ndim, const f64 *data, bool requires_grad) {
    // params and inputs live in perm, so graph_reset never frees them
    Tensor *t = perm_alloc(sizeof(Tensor));
    tensor_set_shape(t, shape, ndim);
    i64 bytes = t->size * (i64)sizeof(f64);
    t->data = perm_alloc(bytes);
    t->grad = requires_grad ? perm_alloc(bytes) : NULL;
    t->requires_grad = requires_grad;
    if (data != NULL) {
        memcpy(t->data, data, (usize)bytes);
    }
    return t;
}
