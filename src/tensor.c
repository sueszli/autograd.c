#include "tensor.h"
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --------------------------------------------------------------------------
// Memory Helpers
// --------------------------------------------------------------------------

// Ensure 64-byte alignment for SIMD operations
static void *safe_aligned_alloc(uint64_t size_bytes) {
    const size_t alignment = 64;
    size_t s_bytes = (size_t)size_bytes;
    // aligned_alloc requires size to be a multiple of alignment
    if (s_bytes % alignment != 0) {
        s_bytes = (s_bytes / alignment + 1) * alignment;
    }
    void *ptr = aligned_alloc(alignment, s_bytes);
    assert(ptr != NULL && "aligned_alloc failed: out of memory");
    return ptr;
}

// --------------------------------------------------------------------------
// Shape / Stride Calculation
// --------------------------------------------------------------------------

static uint64_t calculate_size(const uint64_t *shape, uint64_t ndim) {
    if (ndim == 0) {
        return 1;
    }
    assert(shape != NULL);
    uint64_t size = 1;
    for (uint64_t i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

static void calculate_strides(const uint64_t *shape, uint64_t ndim, uint64_t *out_strides) {
    if (ndim == 0) {
        return;
    }
    assert(shape != NULL);
    assert(out_strides != NULL);

    uint64_t stride = 1;
    for (int64_t i = (int64_t)ndim - 1; i >= 0; i--) {
        out_strides[i] = stride;
        stride *= shape[i];
    }
}

// --------------------------------------------------------------------------
// Tensor Creation / Destruction
// --------------------------------------------------------------------------

// cppcheck-suppress staticFunction
Tensor *tensor_create(const float32_t *data, const uint64_t *shape, uint64_t ndim, bool requires_grad) {
    // Assert constraints
    assert(shape != NULL || ndim == 0);

    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    assert(t != NULL && "malloc failed");

    t->ndim = ndim;
    t->requires_grad = requires_grad;
    t->grad = NULL;
    t->shape = NULL;
    t->strides = NULL;

    // Handle scalar (ndim=0) or empty shape
    if (ndim == 0) {
        t->size = 1;
        // Allocate single element, aligned
        t->data = (float32_t *)safe_aligned_alloc(sizeof(float32_t));
        if (data) {
            t->data[0] = data[0];
        } else {
            t->data[0] = 0.0f;
        }
        return t;
    }

    // Allocate shape and strides
    t->shape = (uint64_t *)malloc((size_t)ndim * sizeof(uint64_t));
    assert(t->shape != NULL && "malloc failed");
    memcpy(t->shape, shape, (size_t)ndim * sizeof(uint64_t));

    t->strides = (uint64_t *)malloc((size_t)ndim * sizeof(uint64_t));
    assert(t->strides != NULL && "malloc failed");
    calculate_strides(t->shape, ndim, t->strides);

    t->size = calculate_size(shape, ndim);

    if (t->size == 0) {
        t->data = NULL;
        return t;
    }

    // Allocate data aligned
    t->data = (float32_t *)safe_aligned_alloc(t->size * sizeof(float32_t));

    if (data) {
        memcpy(t->data, data, (size_t)t->size * sizeof(float32_t));
    } else {
        memset(t->data, 0, (size_t)t->size * sizeof(float32_t));
    }

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

// --------------------------------------------------------------------------
// Broadcasting
// --------------------------------------------------------------------------

/*
 * broadcasting logic:
 * aligns dimensions from the right.
 * compatible if dimensions are equal or one of them is 1.
 *
 * shape_a: [   3, 1]
 * shape_b: [2, 1, 5]
 *          ^  ^  ^
 * out:     [2, 3, 5]
 */
static bool broadcast_shapes(const uint64_t *shape_a, uint64_t ndim_a, const uint64_t *shape_b, uint64_t ndim_b, uint64_t *out_shape, uint64_t *out_ndim) {
    assert(out_shape != NULL);
    assert(out_ndim != NULL);

    uint64_t max_ndim = (ndim_a > ndim_b) ? ndim_a : ndim_b;
    *out_ndim = max_ndim;

    int64_t idx_a = (int64_t)ndim_a - 1;
    int64_t idx_b = (int64_t)ndim_b - 1;
    int64_t idx_out = (int64_t)max_ndim - 1;

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

// --------------------------------------------------------------------------
// Arithmetic
// --------------------------------------------------------------------------

typedef float32_t (*binary_op_t)(float32_t, float32_t);

static float32_t op_add(float32_t a, float32_t b) { return a + b; }
static float32_t op_sub(float32_t a, float32_t b) { return a - b; }
static float32_t op_mul(float32_t a, float32_t b) { return a * b; }
static float32_t op_div(float32_t a, float32_t b) { return a / b; }

/*
 * iterate over the output tensor and map indices back to input tensors
 * considering broadcasting rules.
 *
 * Example:
 * A: (3, 1) -> strides (1, 1)
 * B: (1, 5) -> strides (5, 1)
 * Out: (3, 5) -> strides (5, 1)
 *
 * Index i in Out -> (d0, d1)
 * A index: d0 * stride_a[0] + 0 * stride_a[1]  (since dim 1 is 1)
 * B index: 0 * stride_b[0] + d1 * stride_b[1]  (since dim 0 is 1)
 */
static Tensor *tensor_binary_op(Tensor *a, Tensor *b, binary_op_t op) {
    assert(a != NULL);
    assert(b != NULL);
    assert(op != NULL);

    uint64_t out_shape[32]; // Hard limit of 32 dims should be enough
    uint64_t out_ndim;

    if (!broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, out_shape, &out_ndim)) {
        assert(false && "Shapes cannot be broadcasted");
    }

    Tensor *result = tensor_zeros(out_shape, out_ndim, a->requires_grad || b->requires_grad);

    uint64_t *indices = (uint64_t *)calloc((size_t)out_ndim, sizeof(uint64_t));
    assert(indices != NULL && "calloc failed");

    for (uint64_t i = 0; i < result->size; i++) {
        // unravel index
        uint64_t temp = i;
        for (int64_t d = (int64_t)out_ndim - 1; d >= 0; d--) {
            indices[d] = temp % out_shape[d];
            temp /= out_shape[d];
        }

        uint64_t offset_a = 0;
        for (uint64_t d = 0; d < a->ndim; d++) {
            // map out_dim index to a_dim index
            // align right
            uint64_t result_dim_idx = d + (out_ndim - a->ndim);
            // if a dimension is 1, index is always 0 (broadcast)
            uint64_t idx = (a->shape[d] == 1) ? 0 : indices[result_dim_idx];
            offset_a += idx * a->strides[d];
        }

        uint64_t offset_b = 0;
        for (uint64_t d = 0; d < b->ndim; d++) {
            uint64_t result_dim_idx = d + (out_ndim - b->ndim);
            uint64_t idx = (b->shape[d] == 1) ? 0 : indices[result_dim_idx];
            offset_b += idx * b->strides[d];
        }

        result->data[i] = op(a->data[offset_a], b->data[offset_b]);
    }

    free(indices);
    return result;
}

Tensor *tensor_add(Tensor *a, Tensor *b) { return tensor_binary_op(a, b, op_add); }
Tensor *tensor_sub(Tensor *a, Tensor *b) { return tensor_binary_op(a, b, op_sub); }
Tensor *tensor_mul(Tensor *a, Tensor *b) { return tensor_binary_op(a, b, op_mul); }
Tensor *tensor_div(Tensor *a, Tensor *b) { return tensor_binary_op(a, b, op_div); }

/*
 * Matrix Multiplication (2D only)
 *
 * A: (M, K)
 * B: (K, N)
 * Out: (M, N)
 *
 *        [ b00 b01 ... b0N ]
 *        [ b10 b11 ... b1N ]
 *        [ ...             ]
 *        [ bK0 bK1 ... bKN ]
 *
 * [ a00 ... a0K ] -> [ r00 ... r0N ]
 * [ ...         ]    [ ...         ]
 * [ aM0 ... aMK ]    [ rM0 ... rMN ]
 *
 * r_ij = sum_k (a_ik * b_kj)
 */
Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    assert(a != NULL);
    assert(b != NULL);

    // Validate shapes
    assert(a->ndim >= 1 && b->ndim >= 1 && "Matmul requires at least 1D tensors");
    assert(a->ndim == 2 && b->ndim == 2 && "Only 2D matmul supported");
    assert(a->shape[1] == b->shape[0] && "Inner dimensions must match");

    uint64_t M = a->shape[0];
    uint64_t K = a->shape[1];
    uint64_t N = b->shape[1];

    const uint64_t out_shape[] = {M, N};
    Tensor *result = tensor_zeros(out_shape, 2, a->requires_grad || b->requires_grad);

    // Naive matrix multiplication O(M*N*K)
    for (uint64_t i = 0; i < M; i++) {
        for (uint64_t j = 0; j < N; j++) {
            float32_t sum = 0.0f;
            for (uint64_t k = 0; k < K; k++) {
                sum += a->data[i * a->strides[0] + k * a->strides[1]] * b->data[k * b->strides[0] + j * b->strides[1]];
            }
            result->data[i * result->strides[0] + j * result->strides[1]] = sum;
        }
    }

    return result;
}

// --------------------------------------------------------------------------
// Shape Manipulation
// --------------------------------------------------------------------------

Tensor *tensor_reshape(const Tensor *t, const int64_t *new_shape, uint64_t new_ndim) {
    assert(t != NULL);
    assert(new_shape != NULL);

    uint64_t new_size = 1;
    int64_t unknown_idx = -1;

    // Validate new shape and handle -1
    for (uint64_t i = 0; i < new_ndim; i++) {
        if (new_shape[i] == -1) {
            assert(unknown_idx == -1 && "Only one dimension can be -1");
            unknown_idx = (int64_t)i;
        } else {
            assert(new_shape[i] >= 0 && "Dimension cannot be negative (except -1)");
            new_size *= (uint64_t)new_shape[i];
        }
    }

    if (unknown_idx != -1) {
        assert(t->size % new_size == 0 && "Invalid shape (cannot infer dimension)");
    } else {
        assert(new_size == t->size && "Total elements must match");
    }

    uint64_t *resolved_shape = (uint64_t *)malloc((size_t)new_ndim * sizeof(uint64_t));
    assert(resolved_shape != NULL && "malloc failed");

    for (uint64_t i = 0; i < new_ndim; i++) {
        if ((int64_t)i == unknown_idx) {
            resolved_shape[i] = t->size / new_size;
        } else {
            resolved_shape[i] = (uint64_t)new_shape[i];
        }
    }

    Tensor *result = tensor_create(t->data, resolved_shape, new_ndim, t->requires_grad);
    free(resolved_shape);

    return result;
}

/*
 * Tensor Transpose
 * ----------------
 * Swaps two dimensions.
 *
 * Example: (2, 3) -> (3, 2)
 *
 * T:
 * [[1, 2, 3],
 *  [4, 5, 6]]
 *
 * T.T:
 * [[1, 4],
 *  [2, 5],
 *  [3, 6]]
 *
 * Logic:
 * New Shape: swap(shape[dim0], shape[dim1])
 * Iterate result indices. Map to input indices by swapping dim0 and dim1.
 */
Tensor *tensor_transpose(Tensor *t, uint64_t dim0, uint64_t dim1) {
    assert(t != NULL);

    if (t->ndim < 2) {
        return tensor_create(t->data, t->shape, t->ndim, t->requires_grad);
    }

    assert(dim0 < t->ndim && "Dimension 0 out of bounds");
    assert(dim1 < t->ndim && "Dimension 1 out of bounds");

    uint64_t *new_shape = (uint64_t *)malloc((size_t)t->ndim * sizeof(uint64_t));
    assert(new_shape != NULL && "malloc failed");
    memcpy(new_shape, t->shape, (size_t)t->ndim * sizeof(uint64_t));

    // Swap dimensions in shape
    uint64_t temp = new_shape[dim0];
    new_shape[dim0] = new_shape[dim1];
    new_shape[dim1] = temp;

    Tensor *result = tensor_zeros(new_shape, t->ndim, t->requires_grad);
    free(new_shape);

    uint64_t *indices = (uint64_t *)calloc((size_t)t->ndim, sizeof(uint64_t));
    assert(indices != NULL && "calloc failed");

    // Iterate over result and map to input
    for (uint64_t i = 0; i < result->size; i++) {
        uint64_t temp_i = i;
        for (int64_t d = (int64_t)t->ndim - 1; d >= 0; d--) {
            indices[d] = temp_i % result->shape[d];
            temp_i /= result->shape[d];
        }

        uint64_t offset = 0;
        for (uint64_t d = 0; d < t->ndim; d++) {
            // For input, the indices are swapped compared to output at dim0/dim1
            uint64_t idx_val = indices[d];
            if (d == dim0)
                idx_val = indices[dim1];
            else if (d == dim1)
                idx_val = indices[dim0];

            offset += idx_val * t->strides[d];
        }

        result->data[i] = t->data[offset];
    }
    free(indices);

    return result;
}

// --------------------------------------------------------------------------
// Reductions
// --------------------------------------------------------------------------

static void resolve_axis(uint64_t ndim, int64_t axis, int64_t *out_axis) {
    if (axis < 0) {
        axis += (int64_t)ndim;
    }
    assert(axis >= 0 && axis < (int64_t)ndim && "Axis out of bounds");
    *out_axis = axis;
}

static void calculate_reduction_shape(const Tensor *t, int64_t axis, bool keepdims, uint64_t **out_shape, uint64_t *out_ndim) {
    *out_ndim = keepdims ? t->ndim : t->ndim - 1;
    *out_shape = NULL;

    if (*out_ndim > 0) {
        *out_shape = (uint64_t *)malloc((size_t)(*out_ndim) * sizeof(uint64_t));
        assert(*out_shape != NULL && "malloc failed");
    }

    if (keepdims) {
        for (uint64_t i = 0; i < t->ndim; i++) {
            (*out_shape)[i] = ((int64_t)i == axis) ? 1 : t->shape[i];
        }
    } else {
        uint64_t k = 0;
        for (uint64_t i = 0; i < t->ndim; i++) {
            if ((int64_t)i != axis) {
                (*out_shape)[k++] = t->shape[i];
            }
        }
    }
}

static uint64_t get_reduction_base_offset(const Tensor *t, const uint64_t *indices, int64_t axis, bool keepdims) {
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
        }

        base_offset += idx_val * t->strides[d];

        if (!keepdims) {
            k++;
        }
    }
    return base_offset;
}

// cppcheck-suppress staticFunction
Tensor *tensor_sum(Tensor *t, int64_t axis, bool keepdims) {
    assert(t != NULL);
    resolve_axis(t->ndim, axis, &axis);

    uint64_t *new_shape;
    uint64_t new_ndim;
    calculate_reduction_shape(t, axis, keepdims, &new_shape, &new_ndim);

    Tensor *result = tensor_zeros(new_shape, new_ndim, t->requires_grad);
    if (new_shape)
        free(new_shape);

    uint64_t *indices = (new_ndim > 0) ? (uint64_t *)calloc((size_t)new_ndim, sizeof(uint64_t)) : NULL;
    if (new_ndim > 0)
        assert(indices != NULL && "calloc failed");

    for (uint64_t i = 0; i < result->size; i++) {
        // Unravel result index
        if (new_ndim > 0) {
            uint64_t temp_i = i;
            for (int64_t d = (int64_t)new_ndim - 1; d >= 0; d--) {
                indices[d] = temp_i % result->shape[d];
                temp_i /= result->shape[d];
            }
        }

        uint64_t base_offset = get_reduction_base_offset(t, indices, axis, keepdims);

        // Reduce along axis
        float32_t sum = 0.0f;
        uint64_t axis_dim = (t->shape) ? t->shape[axis] : 1;
        uint64_t axis_stride = t->strides[axis];

        for (uint64_t j = 0; j < axis_dim; j++) {
            sum += t->data[base_offset + j * axis_stride];
        }
        result->data[i] = sum;
    }

    if (indices)
        free(indices);
    return result;
}

Tensor *tensor_mean(Tensor *t, int64_t axis, bool keepdims) {
    assert(t != NULL);
    resolve_axis(t->ndim, axis, &axis); // Just to validate, tensor_sum does it again but that's fine

    Tensor *sum = tensor_sum(t, axis, keepdims);

    uint64_t n = (t->shape) ? t->shape[axis] : 1;
    float32_t scale = 1.0f / (float32_t)n;

    for (uint64_t i = 0; i < sum->size; i++) {
        sum->data[i] *= scale;
    }

    return sum;
}

Tensor *tensor_max(Tensor *t, int64_t axis, bool keepdims) {
    assert(t != NULL);
    resolve_axis(t->ndim, axis, &axis);

    uint64_t *new_shape;
    uint64_t new_ndim;
    calculate_reduction_shape(t, axis, keepdims, &new_shape, &new_ndim);

    Tensor *result = tensor_zeros(new_shape, new_ndim, t->requires_grad);
    if (new_shape)
        free(new_shape);

    uint64_t *indices = (new_ndim > 0) ? (uint64_t *)calloc((size_t)new_ndim, sizeof(uint64_t)) : NULL;
    if (new_ndim > 0)
        assert(indices != NULL && "calloc failed");

    for (uint64_t i = 0; i < result->size; i++) {
        if (new_ndim > 0) {
            uint64_t temp_i = i;
            for (int64_t d = (int64_t)new_ndim - 1; d >= 0; d--) {
                indices[d] = temp_i % result->shape[d];
                temp_i /= result->shape[d];
            }
        }

        uint64_t base_offset = get_reduction_base_offset(t, indices, axis, keepdims);

        float32_t max_val = -INFINITY;
        uint64_t axis_dim = (t->shape) ? t->shape[axis] : 1;
        uint64_t axis_stride = t->strides[axis];

        if (axis_dim > 0) {
            max_val = t->data[base_offset];
            for (uint64_t j = 1; j < axis_dim; j++) {
                float32_t val = t->data[base_offset + j * axis_stride];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        result->data[i] = max_val;
    }

    if (indices)
        free(indices);
    return result;
}

// --------------------------------------------------------------------------
// Utils
// --------------------------------------------------------------------------

static void tensor_print_recursive(Tensor *t, uint64_t dim, uint64_t offset, uint64_t indent) {
    if (dim == t->ndim) {
        printf("%f", t->data[offset]);
        return;
    }

    if (dim == t->ndim - 1) {
        printf("[");
        for (uint64_t i = 0; i < t->shape[dim]; i++) {
            printf("%f", t->data[offset + i * t->strides[dim]]);
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
        // use explicit fixed width type for constant
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

Tensor *tensor_get(Tensor *t, const uint64_t *indices) {
    assert(t != NULL);
    assert(indices != NULL);

    uint64_t offset = 0;
    if (t->ndim > 0) {
        for (uint64_t i = 0; i < t->ndim; i++) {
            assert(indices[i] < t->shape[i] && "Index out of bounds");
            offset += indices[i] * t->strides[i];
        }
    }

    Tensor *res = tensor_create(&t->data[offset], NULL, 0, t->requires_grad);
    return res;
}
