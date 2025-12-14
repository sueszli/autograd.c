#include "tensor.h"
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NDIM 32
#define MAX_TENSOR_SIZE (UINT64_MAX / sizeof(float32_t))

//
// memory management
//

#define CACHELINE_SIZE 64
_Static_assert(CACHELINE_SIZE >= sizeof(float32_t), "cacheline_size must be at least 4 bytes");
_Static_assert((CACHELINE_SIZE & (CACHELINE_SIZE - 1)) == 0, "cacheline_size must be power of 2");

static void *safe_aligned_alloc(uint64_t size_bytes) {
    assert(size_bytes <= MAX_TENSOR_SIZE * sizeof(float32_t));

    size_t s_bytes = (size_t)size_bytes;
    if (s_bytes % CACHELINE_SIZE != 0) {
        s_bytes = (s_bytes / CACHELINE_SIZE + 1) * CACHELINE_SIZE;
    }
    void *ptr = aligned_alloc(CACHELINE_SIZE, s_bytes);
    assert(ptr != NULL && "aligned_alloc failed: out of memory");
    assert((uintptr_t)ptr % CACHELINE_SIZE == 0 && "allocated pointer is not properly aligned");
    return ptr;
}

static uint64_t get_size(const uint64_t *shape, uint64_t ndim) {
    assert(ndim <= MAX_NDIM);

    // scalar
    if (ndim == 0) {
        return 1;
    }
    assert(shape != NULL);

    // product of dimensions
    uint64_t size = 1;
    for (uint64_t i = 0; i < ndim; i++) {
        assert(shape[i] == 0 || size <= MAX_TENSOR_SIZE / shape[i]);
        size *= shape[i];
    }
    assert(size <= MAX_TENSOR_SIZE && "tensor size exceeds maximum");
    return size;
}

/*
 * strides: how many elements to skip in flat memory to move 1 step along each dimension.
 * converts multi-dim index to linear offset: `offset = sum_i (index[i] * strides[i])`
 *
 * example:
 *
 * shape:   [2, 3]  (2 rows, 3 cols)
 *
 * memory:  [a, b, c, d, e, f]
 *
 * logical: [[a, b, c],    row 0
 *           [d, e, f]]    row 1
 *
 * algorithm (iterate backward through dimensions):
 *     i=1: strides[1] = 1   (within a row, move 1 elem)
 *         stride = 1 * 3 = 3
 *     i=0: strides[0] = 3   (between rows, move 3 elems)
 *         stride = 3 * 2 = 6
 *
 * result: strides = [3, 1]
 *
 * access examples:
 *    element[row=1, col=2]: offset = 1*3 + 2*1 = 5 -> data[5] = f
 */
static uint64_t *get_strides(const uint64_t *shape, uint64_t ndim) {
    assert(ndim <= MAX_NDIM);

    if (ndim == 0) {
        return NULL;
    }
    assert(shape != NULL);

    uint64_t *strides = (uint64_t *)malloc((size_t)ndim * sizeof(uint64_t));
    assert(strides != NULL && "malloc failed");

    uint64_t stride = 1;
    for (int64_t i = (int64_t)ndim - 1; i >= 0; i--) {
        strides[i] = stride;
        if (shape[i] && stride > UINT64_MAX / shape[i]) {
            free(strides);
            assert(false && "stride calculation overflow");
        }
        stride *= shape[i];
    }

    return strides;
}

// cppcheck-suppress staticFunction
Tensor *tensor_create(const float32_t *data, const uint64_t *shape, uint64_t ndim, bool requires_grad) {
    assert(ndim <= MAX_NDIM);
    assert(shape != NULL || ndim == 0);

    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    assert(t != NULL && "malloc failed");

    t->ndim = ndim;
    t->requires_grad = requires_grad;
    t->grad = NULL;
    t->shape = NULL;
    t->strides = NULL;

    // scalar
    if (ndim == 0) {
        t->size = 1;
        t->data = (float32_t *)safe_aligned_alloc(sizeof(float32_t));
        assert((uintptr_t)t->data % CACHELINE_SIZE == 0 && "data is not properly aligned");
        if (data) {
            t->data[0] = data[0];
        } else {
            t->data[0] = 0.0f;
        }
        assert(t->ndim == 0);
        assert(t->size == 1);
        assert(t->data != NULL);
        return t;
    }

    t->shape = (uint64_t *)malloc((size_t)ndim * sizeof(uint64_t));
    assert(t->shape != NULL && "malloc failed");
    memcpy(t->shape, shape, (size_t)ndim * sizeof(uint64_t));

    t->strides = get_strides(t->shape, ndim);

    t->size = get_size(shape, ndim);

    // zero-size
    if (t->size == 0) {
        t->data = NULL;
        assert(t->ndim == ndim);
        assert(t->size == 0);
        assert(t->data == NULL);
        return t;
    }

    // data allocation must be aligned
    t->data = (float32_t *)safe_aligned_alloc(t->size * sizeof(float32_t));
    assert((uintptr_t)t->data % CACHELINE_SIZE == 0 && "data is not properly aligned");
    if (data) {
        memcpy(t->data, data, (size_t)t->size * sizeof(float32_t));
    } else {
        memset(t->data, 0, (size_t)t->size * sizeof(float32_t));
    }

    assert(t->ndim == ndim);
    assert(t->size == get_size(shape, ndim));
    assert(t->data != NULL || t->size == 0);
    return t;
}

// cppcheck-suppress staticFunction
Tensor *tensor_zeros(const uint64_t *shape, uint64_t ndim, bool requires_grad) { return tensor_create(NULL, shape, ndim, requires_grad); }

void tensor_free(Tensor *t) {
    if (!t) {
        return;
    }
    if (t->data) {
        free(t->data);
    }
    if (t->shape) {
        free(t->shape);
    }
    if (t->strides) {
        free(t->strides);
    }
    if (t->grad) {
        tensor_free(t->grad);
    }
    free(t);
}

//
// arithmetic
//

/*
 * aligns dimensions from the right.
 * compatible if dimensions are equal or one of them is 1.
 *
 * shape_a: [   3, 1]
 * shape_b: [2, 1, 5]
 *           ^  ^  ^
 * out:     [2, 3, 5]
 */
static bool broadcast_shapes_mut(const uint64_t *shape_a, uint64_t ndim_a, const uint64_t *shape_b, uint64_t ndim_b, uint64_t *out_shape, uint64_t *out_ndim) {
    assert(ndim_a <= MAX_NDIM);
    assert(ndim_b <= MAX_NDIM);
    assert(out_shape != NULL);
    assert(out_ndim != NULL);

    uint64_t max_ndim = (ndim_a > ndim_b) ? ndim_a : ndim_b;
    assert(max_ndim <= MAX_NDIM);
    *out_ndim = max_ndim;

    int64_t idx_a = (int64_t)ndim_a - 1;
    int64_t idx_b = (int64_t)ndim_b - 1;
    int64_t idx_out = (int64_t)max_ndim - 1;

    assert(idx_out < (int64_t)MAX_NDIM);
    while (idx_out >= 0) {
        uint64_t dim_a = (idx_a >= 0 && shape_a) ? shape_a[idx_a] : 1;
        uint64_t dim_b = (idx_b >= 0 && shape_b) ? shape_b[idx_b] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }

        out_shape[idx_out] = (dim_a > dim_b) ? dim_a : dim_b;

        idx_a--;
        idx_b--;
        idx_out--;
    }
    return true;
}

/*
 * converts a linear offset to multi-dimensional indices.
 * mutates out_multidim array.
 *
 * example:
 *
 * shape:   [2, 3]  (2 rows, 3 cols)
 *
 * memory:  [a, b, c, d, e, f]
 *
 * logical: [[a, b, c],    row 0
 *           [d, e, f]]    row 1
 *
 * algorithm (right-to-left):
 *   given: lin=4
 *
 *   d=1 (rightmost/col):  4 % 3 = 1  -> col 1
 *                         4 / 3 = 1  -> carry to next dimension
 *
 *   d=0 (leftmost/row):   1 % 2 = 1  -> row 1
 *                         1 / 2 = 0  -> done
 *
 *   result: [1, 1] -> element 'e'
 */
static void linear_to_multidim_mut(uint64_t lin, const uint64_t *shape, uint64_t ndim, uint64_t *out_multidim) {
    assert(shape != NULL || ndim == 0);
    assert(out_multidim != NULL || ndim == 0);
    assert(ndim <= MAX_NDIM);

    uint64_t carry = lin;
    for (int64_t d = (int64_t)ndim - 1; d >= 0; d--) {
        out_multidim[d] = carry % shape[d];
        carry /= shape[d];
    }
}

/*
 * converts multi-dimensional coordinates to a linear memory offset,
 *
 * example: requesting element at [1, 2] from a tensor
 *
 * shape:   [2, 3]  (2 rows, 3 cols)
 *
 * memory:  [a, b, c]
 *
 * is equivalent to:
 *          [[a, b, c],   // row 0
 *           [a, b, c]]   // row 1 (implicit broadcast)
 *
 *          because the first dimension's size is 1, it behaves as if its shape
 *          were [X, 3] for any X >= 1. implicitly broadcasting.
 *
 * calculation (right-aligned dimensions):
 *
 *   - dimension 0 (rows):
 *     - source size is 1. target requests 1.
 *     - rule: source dimension size is 1 => broadcast! use index 0.
 *     - offset += 0 * stride[0] (3) = 0
 *
 *   - dimension 1 (columns):
 *     - source size is 3. target requests 2.
 *     - rule: source dimension size > 1 => no broadcast. use index 2.
 *     - offset += 2 * stride[1] (1) = 2
 *
 * result: offset = 2 (value 'C').
 */
static uint64_t multidim_to_linear(const uint64_t *target, uint64_t target_ndim, const uint64_t *shape, uint64_t ndim, const uint64_t *strides) {
    assert(target != NULL || target_ndim == 0);
    assert(shape != NULL || ndim == 0);
    assert(strides != NULL || ndim == 0);
    assert(target_ndim >= ndim);
    assert(ndim <= MAX_NDIM);

    uint64_t offset = 0;
    for (uint64_t d = 0; d < ndim; d++) {
        uint64_t target_dim = d + (target_ndim - ndim); // align right
        uint64_t idx = (shape[d] == 1) ? 0 : target[target_dim];
        offset += idx * strides[d];
    }
    return offset;
}

typedef float32_t (*binary_op_t)(float32_t, float32_t);

static float32_t op_add(float32_t a, float32_t b) { return a + b; }
static float32_t op_sub(float32_t a, float32_t b) { return a - b; }
static float32_t op_mul(float32_t a, float32_t b) { return a * b; }
static float32_t op_div(float32_t a, float32_t b) { return a / b; }

static Tensor *tensor_binary_op(Tensor *a, Tensor *b, binary_op_t op) {
    assert(a != NULL);
    assert(b != NULL);
    assert(op != NULL);
    assert(a->data != NULL || a->size == 0);
    assert(b->data != NULL || b->size == 0);
    if (a->size > 0) {
        assert((uintptr_t)a->data % CACHELINE_SIZE == 0 && "a->data is not properly aligned");
    }
    if (b->size > 0) {
        assert((uintptr_t)b->data % CACHELINE_SIZE == 0 && "b->data is not properly aligned");
    }

    uint64_t out_shape[MAX_NDIM];
    uint64_t out_ndim;
    if (!broadcast_shapes_mut(a->shape, a->ndim, b->shape, b->ndim, out_shape, &out_ndim)) {
        assert(false && "shapes cannot be broadcasted");
    }
    assert(out_ndim <= MAX_NDIM);
    Tensor *out_tensor = tensor_zeros(out_shape, out_ndim, a->requires_grad || b->requires_grad);

    // curr = current position in output tensor as multidim indices
    uint64_t *curr = (uint64_t *)calloc((size_t)out_ndim, sizeof(uint64_t));
    assert(curr != NULL && "calloc failed");

    // i = current position in output tensor as linear index
    for (uint64_t i = 0; i < out_tensor->size; i++) {
        // convert i to curr (mutates curr array)
        linear_to_multidim_mut(i, out_shape, out_ndim, curr);

        uint64_t offset_a = multidim_to_linear(curr, out_ndim, a->shape, a->ndim, a->strides);
        assert(offset_a < a->size && "offset_a out of bounds");

        uint64_t offset_b = multidim_to_linear(curr, out_ndim, b->shape, b->ndim, b->strides);
        assert(offset_b < b->size && "offset_b out of bounds");

        out_tensor->data[i] = op(a->data[offset_a], b->data[offset_b]);
    }

    free(curr);

    assert(out_tensor != NULL);
    assert(out_tensor->ndim == out_ndim);
    assert(out_tensor->data != NULL || out_tensor->size == 0);
    return out_tensor;
}

Tensor *tensor_add(Tensor *a, Tensor *b) { return tensor_binary_op(a, b, op_add); }
Tensor *tensor_sub(Tensor *a, Tensor *b) { return tensor_binary_op(a, b, op_sub); }
Tensor *tensor_mul(Tensor *a, Tensor *b) { return tensor_binary_op(a, b, op_mul); }
Tensor *tensor_div(Tensor *a, Tensor *b) { return tensor_binary_op(a, b, op_div); }

Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    assert(a != NULL);
    assert(b != NULL);
    assert(a->data != NULL);
    assert(b->data != NULL);
    assert((uintptr_t)a->data % CACHELINE_SIZE == 0 && "a->data is not properly aligned");
    assert((uintptr_t)b->data % CACHELINE_SIZE == 0 && "b->data is not properly aligned");
    assert(a->ndim >= 1 && b->ndim >= 1 && "matmul requires at least 1D tensors");
    assert(a->ndim == 2 && b->ndim == 2 && "only 2D matmul supported");
    assert(a->shape[1] == b->shape[0] && "inner dimensions must match");

    uint64_t M = a->shape[0];
    uint64_t K = a->shape[1];
    uint64_t N = b->shape[1];

    assert(M <= MAX_TENSOR_SIZE);
    assert(K <= MAX_TENSOR_SIZE);
    assert(N <= MAX_TENSOR_SIZE);

    const uint64_t out_shape[] = {M, N};
    Tensor *result = tensor_zeros(out_shape, 2, a->requires_grad || b->requires_grad);

    // naive algorithm
    for (uint64_t i = 0; i < M; i++) {
        for (uint64_t j = 0; j < N; j++) {
            float32_t sum = 0.0f;
            for (uint64_t k = 0; k < K; k++) {
                uint64_t a_offset = i * a->strides[0] + k * a->strides[1];
                uint64_t b_offset = k * b->strides[0] + j * b->strides[1];
                assert(a_offset < a->size && "a_offset out of bounds");
                assert(b_offset < b->size && "b_offset out of bounds");
                sum += a->data[a_offset] * b->data[b_offset];
            }
            uint64_t result_offset = i * result->strides[0] + j * result->strides[1];
            assert(result_offset < result->size && "result_offset out of bounds");
            result->data[result_offset] = sum;
        }
    }

    assert(result != NULL);
    assert(result->ndim == 2);
    assert(result->shape[0] == M);
    assert(result->shape[1] == N);
    return result;
}

//
// shape manipulation
//

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
Tensor *tensor_transpose(Tensor *t, uint64_t dim0, uint64_t dim1) {
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

//
// reductions
//

/**
 * calculates the output shape and ndim for a tensor reduction.
 *
 * @param t          input tensor.
 * @param dim_idx       index of the dimension to reduce.
 * @param keepdims   if true, maintains reduced dimension with size 1. otherwise, removes it.
 * @param out_shape  [output] pointer to newly allocated array for the resulting shape. caller must free.
 * @param out_ndim   [output] number of dimensions in the resulting shape.
 *
 * example:
 *   input tensor shape: [2, 3]
 *   reduce dim_idx: 0 (first dimension)
 *
 *   if keepdims = false: output shape: [3]
 *
 *   if keepdims = true: output shape: [1, 3]
 */
static void get_reduction_shape_mut(const Tensor *t, int64_t dim_idx, bool keepdims, uint64_t **out_shape, uint64_t *out_ndim) {
    assert(t != NULL);
    assert(t->ndim <= MAX_NDIM);

    *out_ndim = keepdims ? t->ndim : t->ndim - 1;
    assert(*out_ndim <= MAX_NDIM);
    *out_shape = NULL;

    if (*out_ndim > 0) {
        *out_shape = (uint64_t *)malloc((size_t)(*out_ndim) * sizeof(uint64_t));
        assert(*out_shape != NULL && "malloc failed");
    }

    if (keepdims) {
        // ndim stays the same
        for (uint64_t i = 0; i < t->ndim; i++) {
            (*out_shape)[i] = ((int64_t)i == dim_idx) ? 1 : t->shape[i];
        }
    } else {
        // ndim
        uint64_t k = 0;
        for (uint64_t i = 0; i < t->ndim; i++) {
            if ((int64_t)i != dim_idx) {
                (*out_shape)[k++] = t->shape[i];
            }
        }
    }
}

static uint64_t get_reduction_base_offset(const Tensor *t, const uint64_t *indices, int64_t axis, bool keepdims) {
    assert(t != NULL);

    uint64_t base_offset = 0;
    uint64_t k = 0; // index into 'indices' array (which is result-shaped)

    for (uint64_t d = 0; d < t->ndim; d++) {
        if ((int64_t)d == axis) {
            // reduction axis contributes to offset in the inner loop, not base
            continue;
        }

        uint64_t idx_val = 0;
        if (indices) {
            // if keepdims, result has same ndim, so we use d
            // if !keepdims, result has ndim-1, so we use k
            idx_val = indices[keepdims ? d : k];
            if (t->shape) {
                assert(idx_val < t->shape[d] && "index out of bounds");
            }
        }

        base_offset += idx_val * t->strides[d];

        if (!keepdims) {
            k++;
        }
    }
    assert(base_offset < t->size && "base_offset out of bounds");
    return base_offset;
}

// cppcheck-suppress staticFunction
Tensor *tensor_sum(Tensor *t, int64_t axis, bool keepdims) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);
    if (axis < 0) {
        axis += (int64_t)t->ndim;
    }
    assert(axis >= 0 && axis < (int64_t)t->ndim && "axis out of bounds");

    uint64_t *new_shape;
    uint64_t new_ndim;
    get_reduction_shape_mut(t, axis, keepdims, &new_shape, &new_ndim);

    Tensor *result = tensor_zeros(new_shape, new_ndim, t->requires_grad);
    if (new_shape) {
        free(new_shape);
    }

    uint64_t *curr = (new_ndim > 0) ? (uint64_t *)calloc((size_t)new_ndim, sizeof(uint64_t)) : NULL;
    if (new_ndim > 0) {
        assert(curr != NULL && "calloc failed");
    }

    for (uint64_t i = 0; i < result->size; i++) {
        linear_to_multidim_mut(i, result->shape, new_ndim, curr);

        uint64_t base_offset = get_reduction_base_offset(t, curr, axis, keepdims);

        // Reduce along axis
        float32_t sum = 0.0f;
        uint64_t axis_dim = (t->shape) ? t->shape[axis] : 1;
        assert(axis_dim <= MAX_TENSOR_SIZE && "axis_dim exceeds maximum tensor size");
        uint64_t axis_stride = t->strides[axis];

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

Tensor *tensor_mean(Tensor *t, int64_t axis, bool keepdims) {
    assert(t != NULL);
    if (axis < 0) {
        axis += (int64_t)t->ndim;
    }
    assert(axis >= 0 && axis < (int64_t)t->ndim && "axis out of bounds");

    // mutates the returned tensor in place
    Tensor *sum_mut = tensor_sum(t, axis, keepdims);

    uint64_t n = (t->shape) ? t->shape[axis] : 1;
    assert(n > 0 && "division by zero: axis dimension is 0");
    float32_t scale = 1.0f / (float32_t)n;

    for (uint64_t i = 0; i < sum_mut->size; i++) {
        sum_mut->data[i] *= scale;
    }

    return sum_mut;
}

Tensor *tensor_max(Tensor *t, int64_t axis, bool keepdims) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);
    if (axis < 0) {
        axis += (int64_t)t->ndim;
    }
    assert(axis >= 0 && axis < (int64_t)t->ndim && "axis out of bounds");

    uint64_t *new_shape;
    uint64_t new_ndim;
    get_reduction_shape_mut(t, axis, keepdims, &new_shape, &new_ndim);

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

        uint64_t base_offset = get_reduction_base_offset(t, curr, axis, keepdims);

        float32_t max_val = -INFINITY;
        uint64_t axis_dim = (t->shape) ? t->shape[axis] : 1;
        uint64_t axis_stride = t->strides[axis];

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

//
// utils
//

static void tensor_print_recursive(Tensor *t, uint64_t dim, uint64_t offset, uint64_t indent) {
    assert(dim <= MAX_NDIM && "recursion depth exceeds maximum");
    assert(t != NULL);

    if (dim == t->ndim) {
        assert(offset < t->size && "offset out of bounds");
        printf("%f", t->data[offset]);
        return;
    }

    if (dim == t->ndim - 1) {
        printf("[");
        for (uint64_t i = 0; i < t->shape[dim]; i++) {
            uint64_t data_offset = offset + i * t->strides[dim];
            assert(data_offset < t->size && "offset out of bounds");
            printf("%f", t->data[data_offset]);
            if (i < t->shape[dim] - 1) {
                printf(", ");
            }
        }
        printf("]");
        return;
    }

    printf("[");
    for (uint64_t i = 0; i < t->shape[dim]; i++) {
        if (i > 0) {
            for (uint64_t j = 0; j < indent; j++) {
                printf(" ");
            }
        }
        tensor_print_recursive(t, dim + 1, offset + i * t->strides[dim], indent + 1);

        if (i < t->shape[dim] - 1) {
            printf(",");
            uint64_t newlines = t->ndim - dim - 1;
            for (uint64_t k = 0; k < newlines; k++) {
                printf("\n");
            }
        }
    }
    printf("]");
}

void tensor_print(Tensor *t) {
    if (!t) {
        printf("Tensor(NULL)\n");
        return;
    }
    printf("Tensor(shape=[");
    if (t->shape) {
        for (uint64_t i = 0; i < t->ndim; i++) {
            printf("%" PRIu64 "%s", t->shape[i], i < t->ndim - 1 ? ", " : "");
        }
    }
    printf("], size=%" PRIu64 ", requires_grad=%s)\n", t->size, t->requires_grad ? "true" : "false");

    if (t->data) {
        const uint64_t max_size = 1000;
        if (t->size <= max_size) {
            printf("Data: ");
            tensor_print_recursive(t, 0, 0, 6);
            printf("\n");
        } else {
            printf("Data: ... (size > 1000)\n");
        }
    }
}

// use stride to get offset in flat data array
Tensor *tensor_get(Tensor *t, const uint64_t *indices) {
    assert(t != NULL);
    assert(indices != NULL);
    assert(t->data != NULL || t->size == 0);

    uint64_t offset = 0;
    if (t->ndim > 0) {
        for (uint64_t i = 0; i < t->ndim; i++) {
            assert(indices[i] < t->shape[i] && "idx out of bounds");
            offset += indices[i] * t->strides[i];
        }
    }
    assert(offset < t->size && "offset out of bounds");

    Tensor *res = tensor_create(&t->data[offset], NULL, 0, t->requires_grad);
    return res;
}
