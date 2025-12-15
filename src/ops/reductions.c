#include "ops/reductions.h"
#include "ops/arithmetic.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

/*
 * calculates the output shape and ndim for a tensor reduction
 *
 * example:
 *   input shape:   [2, 3]
 *   reduce dim_idx: 0 (first dimension)
 *
 *   a) keepdims == false -> output shape: [3]
 *   b) keepdims == true  -> output shape: [1, 3]
 */
static void reduction_shapes_mut(const Tensor *t, int64_t dim_idx, bool keepdims, uint64_t **out_shape, uint64_t *out_ndim) {
    assert(t != NULL);
    assert(t->ndim <= MAX_NDIM);
    dim_idx = (dim_idx < 0) ? (dim_idx + (int64_t)t->ndim) : dim_idx; // handle negative indices
    assert(dim_idx >= 0 && dim_idx < (int64_t)t->ndim && "dim_idx out of bounds");

    *out_ndim = keepdims ? t->ndim : t->ndim - 1;
    assert(*out_ndim <= MAX_NDIM);
    *out_shape = NULL;

    if (*out_ndim > 0) {
        *out_shape = (uint64_t *)malloc((size_t)(*out_ndim) * sizeof(uint64_t));
        assert(*out_shape != NULL && "malloc failed");
    }

    if (keepdims) {
        // keep dim_idx, collapse to size 1
        for (uint64_t i = 0; i < t->ndim; i++) {
            (*out_shape)[i] = ((int64_t)i == dim_idx) ? 1 : t->shape[i];
        }
    } else {
        // drop dim_idx entirely
        uint64_t k = 0;
        for (uint64_t i = 0; i < t->ndim; i++) {
            if ((int64_t)i != dim_idx) {
                (*out_shape)[k++] = t->shape[i];
            }
        }
    }
}

// same as multidim_to_linear, but skips the reduced dimension
static uint64_t reduction_multidim_to_linear(const Tensor *t, const uint64_t *multidim, int64_t dim_idx, bool keepdims) {
    assert(t != NULL);
    dim_idx = (dim_idx < 0) ? (dim_idx + (int64_t)t->ndim) : dim_idx;
    assert(dim_idx >= 0 && dim_idx < (int64_t)t->ndim && "dim_idx out of bounds");

    uint64_t offset = 0;
    for (uint64_t d = 0; d < t->ndim; d++) {
        // skip reduced dimension
        if ((int64_t)d == dim_idx) {
            continue;
        }

        assert(multidim != NULL);
        // map d (original dim) to index in multidim (reduced shape)
        uint64_t idx = keepdims ? d : (d > (uint64_t)dim_idx ? d - 1 : d);
        uint64_t idx_val = multidim[idx];
        assert(t->shape != NULL);
        assert(idx_val < t->shape[d] && "index out of bounds");

        offset += idx_val * t->strides[d];
    }
    assert((t->size == 0 || offset < t->size) && "offset out of bounds");
    return offset;
}

/*
 * sums elements along a dimension.
 *
 * example:
 *
 * shape:   [2, 3]
 *
 * logical: [[1, 2, 3],
 *           [4, 5, 6]]
 *
 * operation (sum along dim 0):
 *
 *   a) keepdims = true
 *      shape:  [1, 3]
 *      result: [[5, 7, 9]]
 *
 *   b) keepdims = false
 *      shape:  [3]
 *      result: [5, 7, 9]
 */
Tensor *tensor_sum(const Tensor *t, int64_t dim_idx, bool keepdims) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);
    dim_idx = (dim_idx < 0) ? (dim_idx + (int64_t)t->ndim) : dim_idx;
    assert(dim_idx >= 0 && dim_idx < (int64_t)t->ndim && "dim_idx out of bounds");

    uint64_t *new_shape;
    uint64_t new_ndim;
    reduction_shapes_mut(t, dim_idx, keepdims, &new_shape, &new_ndim);

    Tensor *result = tensor_zeros(new_shape, new_ndim, t->requires_grad);
    if (new_shape) {
        free(new_shape);
    }

    // buffer for current multidim index
    uint64_t *curr = (new_ndim > 0) ? (uint64_t *)calloc((size_t)new_ndim, sizeof(uint64_t)) : NULL;
    if (new_ndim > 0) {
        assert(curr != NULL && "calloc failed");
    }

    for (uint64_t i = 0; i < result->size; i++) {
        linear_to_multidim_mut(i, result->shape, new_ndim, curr);

        uint64_t base_offset = reduction_multidim_to_linear(t, curr, dim_idx, keepdims);

        // sum along axis_dim
        float32_t sum = 0.0f;
        uint64_t axis_dim = (t->shape) ? t->shape[dim_idx] : 1;
        assert(axis_dim <= MAX_TENSOR_SIZE && "axis_dim exceeds maximum tensor size");
        uint64_t axis_stride = t->strides[dim_idx];
        for (uint64_t j = 0; j < axis_dim; j++) {
            uint64_t offset = base_offset + j * axis_stride;
            assert(offset < t->size && "offset out of bounds");
            sum += t->data[offset];
        }
        result->data[i] = sum;
    }

    if (curr) {
        free(curr);
    }
    return result;
}

Tensor *tensor_mean(const Tensor *t, int64_t dim_idx, bool keepdims) {
    assert(t != NULL);
    dim_idx = (dim_idx < 0) ? (dim_idx + (int64_t)t->ndim) : dim_idx;
    assert(dim_idx >= 0 && dim_idx < (int64_t)t->ndim && "dim_idx out of bounds");

    Tensor *sum_mut = tensor_sum(t, dim_idx, keepdims);

    // scale sum by 1/n
    uint64_t n = (t->shape) ? t->shape[dim_idx] : 1;
    assert(n > 0 && "division by zero: axis dimension is 0");

    // create a scalar tensor for scaling to support autograd
    const uint64_t scalar_shape[] = {1};
    Tensor *scale_t = tensor_create(NULL, scalar_shape, 0, false);
    scale_t->data[0] = 1.0f / (float32_t)n;

    Tensor *result = tensor_mul(sum_mut, scale_t);

    tensor_free(scale_t);
    tensor_free(sum_mut);

    return result;
}

Tensor *tensor_max(const Tensor *t, int64_t dim_idx, bool keepdims) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);
    dim_idx = (dim_idx < 0) ? (dim_idx + (int64_t)t->ndim) : dim_idx;
    assert(dim_idx >= 0 && dim_idx < (int64_t)t->ndim && "dim_idx out of bounds");

    uint64_t *new_shape;
    uint64_t new_ndim;
    reduction_shapes_mut(t, dim_idx, keepdims, &new_shape, &new_ndim);

    Tensor *result = tensor_zeros(new_shape, new_ndim, t->requires_grad);
    if (new_shape) {
        free(new_shape);
    }

    uint64_t *curr = (new_ndim > 0) ? (uint64_t *)calloc((size_t)new_ndim, sizeof(uint64_t)) : NULL;
    if (new_ndim > 0) {
        assert(curr != NULL && "calloc failed");
    }

    for (uint64_t i = 0; i < result->size; i++) {
        if (new_ndim > 0) {
            linear_to_multidim_mut(i, result->shape, new_ndim, curr);
        }

        uint64_t base_offset = reduction_multidim_to_linear(t, curr, dim_idx, keepdims);

        float32_t max_val = -INFINITY;
        uint64_t axis_dim = (t->shape) ? t->shape[dim_idx] : 1;
        uint64_t axis_stride = t->strides[dim_idx];

        if (axis_dim > 0) {
            assert(base_offset < t->size && "base_offset out of bounds");
            max_val = t->data[base_offset];
            for (uint64_t j = 1; j < axis_dim; j++) {
                uint64_t offset = base_offset + j * axis_stride;
                assert(offset < t->size && "offset out of bounds");
                float32_t val = t->data[offset];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        result->data[i] = max_val;
    }

    if (curr) {
        free(curr);
    }
    return result;
}
