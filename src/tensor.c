#include "tensor.h"
#include "autograd.h"
#include "ops.h"
#include <assert.h>
#include <string.h>

void set_shape(Tensor *t, const i64 *shape, i64 ndim) {
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

static Tensor *tensor_new(Mem *mem, const i64 *shape, i64 ndim, const f64 *data, bool requires_grad) {
    // params and inputs live in perm, so reset_graph never frees them
    Tensor *t = arena.alloc(&mem->perm, sizeof(Tensor));
    set_shape(t, shape, ndim);
    usize bytes = (usize)t->size * sizeof(f64);
    t->data = arena.alloc(&mem->perm, bytes);
    t->grad = requires_grad ? arena.alloc(&mem->perm, bytes) : NULL;
    t->mem = mem;
    t->requires_grad = requires_grad;
    if (data != NULL) {
        memcpy(t->data, data, bytes);
    }
    return t;
}

const TensorNamespace tensor = {
    .new = tensor_new,
    .backward = autograd_backward,
    .reset_graph = autograd_reset,
    .add = op_add,
    .mul = op_mul,
    .relu = op_relu,
    .matmul = op_matmul,
    .transpose = op_transpose,
    .embedding = op_embedding,
    .masked_fill_upper = op_masked_fill_upper,
    .narrow = op_narrow,
    .cat = op_cat,
    .rms_norm = op_rms_norm,
    .softmax = op_softmax,
    .cross_entropy = op_cross_entropy,
};
