#include "ops/reshapes.h"
#include "utils/aligned_alloc.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

Tensor *tensor_reshape(const Tensor *t, const int64_t *new_shape, uint64_t new_ndim) {
    assert(t != NULL);
    assert(new_shape != NULL);
    assert(t->data != NULL || t->size == 0);
    assert(new_ndim <= MAX_NDIM);

    uint64_t new_size = 1;
    int64_t unknown_idx = -1; // one dimension can be -1 (inferred)

    // validate
    for (uint64_t i = 0; i < new_ndim; i++) {
        if (new_shape[i] == -1) {
            assert(unknown_idx == -1 && "only one dimension can be -1");
            unknown_idx = (int64_t)i;
        } else {
            assert(new_shape[i] >= 0 && "dimension cannot be negative (except -1)");
            new_size *= (uint64_t)new_shape[i];
        }
    }
    if (unknown_idx != -1) {
        assert(t->size % new_size == 0 && "invalid shape (cannot infer dimension)");
    } else {
        assert(new_size == t->size && "total elements must match");
    }

    uint64_t *resolved_shape = (uint64_t *)malloc((size_t)new_ndim * sizeof(uint64_t));
    assert(resolved_shape != NULL && "malloc failed");

    // fill in
    for (uint64_t i = 0; i < new_ndim; i++) {
        if ((int64_t)i == unknown_idx) {
            resolved_shape[i] = t->size / new_size;
        } else {
            resolved_shape[i] = (uint64_t)new_shape[i];
        }
    }

    Tensor *result = tensor_create(t->data, resolved_shape, new_ndim, t->requires_grad);
    free(resolved_shape);
    assert(result != NULL);
    assert(result->size == t->size);

    return result;
}

/*
 * transpose swaps two dimensions
 *
 * example: (2, 3) -> (3, 2)
 *
 *   T:
 *   [[1, 2, 3],
 *    [4, 5, 6]]
 *
 *   T.T:
 *   [[1, 4],
 *    [2, 5],
 *    [3, 6]]
 */
Tensor *tensor_transpose(const Tensor *t, uint64_t dim0, uint64_t dim1) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);
    assert(t->ndim <= MAX_NDIM);
    if (t->size > 0) {
        assert((uintptr_t)t->data % CACHELINE_SIZE == 0 && "data is not properly aligned");
    }

    if (t->ndim < 2) {
        return tensor_create(t->data, t->shape, t->ndim, t->requires_grad);
    }

    assert(dim0 < t->ndim && "dimension 0 out of bounds");
    assert(dim1 < t->ndim && "dimension 1 out of bounds");

    uint64_t *new_shape = (uint64_t *)malloc((size_t)t->ndim * sizeof(uint64_t));
    assert(new_shape != NULL && "malloc failed");
    memcpy(new_shape, t->shape, (size_t)t->ndim * sizeof(uint64_t));

    // swap dims
    uint64_t temp = new_shape[dim0];
    new_shape[dim0] = new_shape[dim1];
    new_shape[dim1] = temp;

    Tensor *result = tensor_zeros(new_shape, t->ndim, t->requires_grad);
    free(new_shape);

    uint64_t *curr = (uint64_t *)calloc((size_t)t->ndim, sizeof(uint64_t));
    assert(curr != NULL && "calloc failed");

    for (uint64_t i = 0; i < result->size; i++) {
        linear_to_multidim_mut(i, result->shape, t->ndim, curr);

        uint64_t offset = 0;
        for (uint64_t d = 0; d < t->ndim; d++) {
            // multidim indices are swapped compared to output at dim0/dim1
            uint64_t idx_val = curr[d];
            if (d == dim0) {
                idx_val = curr[dim1];
            } else if (d == dim1) {
                idx_val = curr[dim0];
            }
            offset += idx_val * t->strides[d];
        }
        assert(offset < t->size && "offset out of bounds");

        result->data[i] = t->data[offset];
    }
    free(curr);
    return result;
}
