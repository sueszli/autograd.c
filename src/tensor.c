#include "tensor.h"
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Helper: calculate_size
 * ----------------------
 * Calculates the total number of elements in a tensor from its shape.
 *
 * Steps:
 * 1. Initialize size to 1.
 * 2. Iterate through each dimension in the shape.
 * 3. Multiply size by the dimension size.
 *
 * Example: shape = [2, 3]
 * Iteration 0: size = 1 * 2 = 2
 * Iteration 1: size = 2 * 3 = 6
 * Result: 6
 */
static uint64_t calculate_size(const uint64_t *shape, uint64_t ndim) {
    assert(shape != NULL || ndim == 0);
    uint64_t size = 1;
    for (uint64_t i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

/*
 * Helper: calculate_strides
 * -------------------------
 * Calculates the strides for a tensor. Strides represent the number of
 * elements to skip in memory to reach the next element in a dimension.
 * For a row-major layout (C-style), the last dimension has a stride of 1.
 *
 * Layout: Row-Major (last dimension is contiguous)
 *
 * Example: shape = [2, 3, 4]
 * Strides calculation (from last to first):
 * - Dim 2 (size 4): stride = 1
 * - Dim 1 (size 3): stride = 1 * 4 = 4
 * - Dim 0 (size 2): stride = 4 * 3 = 12
 * Result: strides = [12, 4, 1]
 *
 * Steps:
 * 1. Initialize stride to 1.
 * 2. Iterate backwards from the last dimension to the first.
 * 3. Assign current stride to the current dimension.
 * 4. Update stride by multiplying with the current dimension size.
 */
static void calculate_strides(const uint64_t *shape, uint64_t ndim, uint64_t *strides) {
    assert(shape != NULL || ndim == 0);
    assert(strides != NULL || ndim == 0);
    uint64_t stride = 1;
    for (int64_t i = (int64_t)ndim - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

/*
 * Function: tensor_create
 * -----------------------
 * Allocates and initializes a new Tensor.
 *
 * Memory Layout:
 * +----------------+
 * | Tensor Struct  |
 * | - data ptr  ---|---> [ float, float, ... ] (contiguous data)
 * | - shape ptr ---|---> [ dim0, dim1, ... ]
 * | - strides ptr -|---> [ stride0, stride1, ... ]
 * | ...            |
 * +----------------+
 *
 * Steps:
 * 1. Validate inputs (assert).
 * 2. Allocate memory for the Tensor struct.
 * 3. Handle scalar/empty shape case (ndim=0):
 *    - Allocate single element data.
 *    - Set shape/strides to NULL.
 * 4. For non-scalars:
 *    - Allocate and copy shape array.
 *    - Allocate and calculate strides array.
 *    - Calculate total size.
 *    - Allocate data array.
 * 5. Initialize data:
 *    - Copy from input `data` if provided.
 *    - Zero-initialize if `data` is NULL.
 * 6. Return the new Tensor pointer.
 */
// cppcheck-suppress staticFunction
Tensor *tensor_create(const float32_t *data, const uint64_t *shape, uint64_t ndim, bool requires_grad) {
    // Assert inputs
    // data can be NULL
    assert(shape != NULL || ndim == 0);
    // ndim can be 0 (scalar)

    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t)
        return NULL;

    t->ndim = ndim;
    t->requires_grad = requires_grad;
    t->grad = NULL;
    t->shape = NULL;
    t->strides = NULL;

    // safety check for empty shape (scalar)
    if (ndim == 0 || !shape) {
        t->ndim = 0;
        t->size = 1;
        t->data = (float32_t *)calloc(1, sizeof(float32_t));
        if (!t->data) {
            free(t);
            return NULL;
        }
        if (data)
            t->data[0] = data[0];

        assert(t->size == 1);
        assert(t->ndim == 0);
        return t;
    }

    t->shape = (uint64_t *)malloc((size_t)ndim * sizeof(uint64_t));
    if (!t->shape) {
        free(t);
        return NULL;
    }
    memcpy(t->shape, shape, (size_t)ndim * sizeof(uint64_t));

    t->strides = (uint64_t *)malloc((size_t)ndim * sizeof(uint64_t));
    if (!t->strides) {
        free(t->shape);
        free(t);
        return NULL;
    }
    calculate_strides(t->shape, ndim, t->strides);

    // determine size
    t->size = calculate_size(shape, ndim);

    // allocate data
    if (t->size == 0) {
        t->data = NULL;
        return t;
    }

    t->data = (float32_t *)malloc((size_t)t->size * sizeof(float32_t));
    if (!t->data) {
        free(t->strides);
        free(t->shape);
        free(t);
        return NULL;
    }

    if (data) {
        memcpy(t->data, data, (size_t)t->size * sizeof(float32_t));
    } else {
        memset(t->data, 0, (size_t)t->size * sizeof(float32_t));
    }

    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);
    return t;
}

/*
 * Function: tensor_zeros
 * ----------------------
 * Creates a tensor filled with zeros.
 * Wrapper around tensor_create with NULL data.
 */
// cppcheck-suppress staticFunction
Tensor *tensor_zeros(const uint64_t *shape, uint64_t ndim, bool requires_grad) {
    assert(shape != NULL || ndim == 0);
    return tensor_create(NULL, shape, ndim, requires_grad);
}

/*
 * Function: tensor_free
 * ---------------------
 * Frees all memory associated with a tensor.
 *
 * Steps:
 * 1. Check if tensor is NULL.
 * 2. Free data array.
 * 3. Free shape array.
 * 4. Free strides array.
 * 5. Recursively free gradient tensor if present.
 * 6. Free the tensor struct itself.
 */
// cppcheck-suppress staticFunction
void tensor_free(Tensor *t) {
    if (!t)
        return;
    if (t->data)
        free(t->data);
    if (t->shape)
        free(t->shape);
    if (t->strides)
        free(t->strides);
    if (t->grad)
        tensor_free(t->grad);
    free(t);
}

/*
 * Helper: broadcast_shapes
 * ------------------------
 * Determines the output shape when broadcasting two shapes together.
 * Broadcasting rules (NumPy style):
 * 1. Align shapes to the right.
 * 2. Dimensions are compatible if they are equal or one of them is 1.
 *
 * Diagram:
 * Shape A:       (2, 3)
 * Shape B:       (   3) -> implicitly (1, 3)
 *                |  |
 * Matches:       No Yes (3==3)
 * Broadcast:     (2, 3)
 *
 * Steps:
 * 1. Determine max rank (max_ndim).
 * 2. Iterate from the last dimension backwards.
 * 3. Get dim_a and dim_b (pad with 1 if out of bounds).
 * 4. Check compatibility (dim_a == dim_b OR dim_a == 1 OR dim_b == 1).
 * 5. Output dim is max(dim_a, dim_b).
 */
static bool broadcast_shapes(const uint64_t *shape_a, uint64_t ndim_a, const uint64_t *shape_b, uint64_t ndim_b, uint64_t *out_shape, uint64_t *out_ndim) {
    assert(shape_a != NULL || ndim_a == 0);
    assert(shape_b != NULL || ndim_b == 0);
    assert(out_shape != NULL);
    assert(out_ndim != NULL);

    uint64_t max_ndim = ndim_a > ndim_b ? ndim_a : ndim_b;
    *out_ndim = max_ndim;

    int64_t idx_a = (int64_t)ndim_a - 1;
    int64_t idx_b = (int64_t)ndim_b - 1;
    int64_t idx_out = (int64_t)max_ndim - 1;

    while (idx_out >= 0) {
        uint64_t dim_a = (idx_a >= 0 && shape_a) ? shape_a[idx_a] : 1;
        uint64_t dim_b = (idx_b >= 0 && shape_b) ? shape_b[idx_b] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1)
            return false;

        out_shape[idx_out] = (dim_a > dim_b) ? dim_a : dim_b;

        idx_a--;
        idx_b--;
        idx_out--;
    }
    return true;
}

//
// arithmetic
//

typedef float32_t (*binary_op_t)(float32_t, float32_t);

static float32_t op_add(float32_t a, float32_t b) { return a + b; }
static float32_t op_sub(float32_t a, float32_t b) { return a - b; }
static float32_t op_mul(float32_t a, float32_t b) { return a * b; }
static float32_t op_div(float32_t a, float32_t b) { return a / b; }

/*
 * Helper: tensor_binary_op
 * ------------------------
 * Performs a binary operation (add, sub, mul, div) element-wise with broadcasting.
 *
 * Process:
 * 1. Determine broadcasted output shape.
 * 2. Create result tensor.
 * 3. Iterate through every element in the result tensor (linear index `i`).
 * 4. Convert linear index `i` -> multidimensional indices `indices[]`.
 * 5. Map `indices[]` back to source indices for A and B, handling broadcasting.
 *    - If dim size is 1, index is 0.
 *    - Otherwise, index matches.
 * 6. Calculate offsets in A and B data arrays using strides.
 * 7. Apply operation and store result.
 *
 * Map Indices Example:
 * Result Shape: (2, 2)
 * A Shape:      (2, 1)
 * Indices:      [1, 1]
 * A Indices:    [1, 0] (since dim 1 is size 1)
 */
static Tensor *tensor_binary_op(Tensor *a, Tensor *b, binary_op_t op) {
    assert(a != NULL);
    assert(b != NULL);
    assert(op != NULL);

    uint64_t out_shape[32];
    uint64_t out_ndim;

    if (!broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, out_shape, &out_ndim)) {
        fprintf(stderr, "error: shapes cannot be broadcasted.\n");
        return NULL;
    }

    Tensor *result = tensor_zeros(out_shape, out_ndim, a->requires_grad || b->requires_grad);
    if (!result)
        return NULL;

    uint64_t *indices = (uint64_t *)calloc((size_t)out_ndim, sizeof(uint64_t));
    if (out_ndim > 0 && !indices) {
        tensor_free(result);
        return NULL;
    }

    for (uint64_t i = 0; i < result->size; i++) {
        if (out_ndim > 0) {
            uint64_t temp = i;
            for (int64_t d = (int64_t)out_ndim - 1; d >= 0; d--) {
                indices[d] = temp % out_shape[d];
                temp /= out_shape[d];
            }
        }

        uint64_t offset_a = 0;
        for (uint64_t d = 0; d < a->ndim; d++) {
            uint64_t result_dim_idx = d + (out_ndim - a->ndim);
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

    if (indices)
        free(indices);

    assert(result != NULL);
    return result;
}

Tensor *tensor_add(Tensor *a, Tensor *b) {
    assert(a != NULL);
    assert(b != NULL);
    return tensor_binary_op(a, b, op_add);
}
Tensor *tensor_sub(Tensor *a, Tensor *b) {
    assert(a != NULL);
    assert(b != NULL);
    return tensor_binary_op(a, b, op_sub);
}
Tensor *tensor_mul(Tensor *a, Tensor *b) {
    assert(a != NULL);
    assert(b != NULL);
    return tensor_binary_op(a, b, op_mul);
}
Tensor *tensor_div(Tensor *a, Tensor *b) {
    assert(a != NULL);
    assert(b != NULL);
    return tensor_binary_op(a, b, op_div);
}

/*
 * Function: tensor_matmul
 * -----------------------
 * Performs matrix multiplication of two 2D tensors.
 *
 * Operation: C = A @ B
 * Shapes:    (M, K) @ (K, N) -> (M, N)
 *
 * Diagram:
 *       [ B00 B01 ]
 *       [ B10 B11 ]
 *       [ B20 B21 ]
 *          (KxN)
 *
 * [ A00 A01 A02 ]  x  [ C00 C01 ]
 * [ A10 A11 A12 ]     [ C10 C11 ]
 *    (MxK)               (MxN)
 *
 * Steps:
 * 1. Validate inputs (must be 2D, inner dimensions match).
 * 2. Create result tensor of shape (M, N).
 * 3. Triple loop implementation:
 *    - Loop i from 0 to M (rows of A)
 *      - Loop j from 0 to N (cols of B)
 *        - Accumulate sum = sum(A[i, k] * B[k, j]) for k in 0..K
 *        - Store sum in C[i, j]
 */
Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    assert(a != NULL);
    assert(b != NULL);

    if (a->ndim < 1 || b->ndim < 1) {
        fprintf(stderr, "error: matmul requires at least 1d tensors.\n");
        return NULL;
    }

    if (a->ndim != 2 || b->ndim != 2) {
        fprintf(stderr, "error: only 2d matmul supported in c implementation for now.\n");
        return NULL;
    }

    if (a->shape[1] != b->shape[0]) {
        fprintf(stderr, "error: inner dimensions must match: %" PRIu64 " != %" PRIu64 "\n", a->shape[1], b->shape[0]);
        return NULL;
    }

    uint64_t M = a->shape[0];
    uint64_t K = a->shape[1];
    uint64_t N = b->shape[1];

    const uint64_t out_shape[] = {M, N};
    Tensor *result = tensor_zeros(out_shape, 2, a->requires_grad || b->requires_grad);
    if (!result)
        return NULL;

    for (uint64_t i = 0; i < M; i++) {
        for (uint64_t j = 0; j < N; j++) {
            float32_t sum = 0;
            for (uint64_t k = 0; k < K; k++) {
                sum += a->data[i * a->strides[0] + k * a->strides[1]] * b->data[k * b->strides[0] + j * b->strides[1]];
            }
            result->data[i * result->strides[0] + j * result->strides[1]] = sum;
        }
    }

    assert(result != NULL);
    return result;
}

//
// shape manipulation
//

/*
 * Function: tensor_reshape
 * ------------------------
 * Returns a new tensor with the same data but a different shape.
 *
 * Rules:
 * - New shape must contain the same total number of elements.
 * - One dimension can be -1 (inferred).
 *
 * Example:
 * Input:  (2, 3), size=6
 * Reshape: (3, 2) -> Valid
 * Reshape: (6)    -> Valid
 * Reshape: (6, -1)-> Inferred to (6, 1)
 *
 * Steps:
 * 1. Validate inputs and calculate new size.
 * 2. Handle inferred dimension (-1):
 *    - Calculate target size from known dimensions.
 *    - Divide total size by target size to find missing dimension.
 * 3. Verify size consistency.
 * 4. Create new tensor sharing the *same data pointer* (wait, code copies data).
 *    - Current implementation creates a NEW tensor with COPIED data.
 *    - It calculates the fully resolved shape.
 *    - Calls tensor_create.
 */
Tensor *tensor_reshape(const Tensor *t, const int64_t *new_shape, uint64_t new_ndim) {
    assert(t != NULL);
    assert(new_shape != NULL);

    uint64_t new_size = 1;
    int64_t unknown_idx = -1;
    for (uint64_t i = 0; i < new_ndim; i++) {
        if (new_shape[i] == -1) {
            if (unknown_idx != -1) {
                fprintf(stderr, "error: only one dimension can be -1.\n");
                return NULL;
            }
            unknown_idx = (int64_t)i;
        } else {
            if (new_shape[i] < 0) {
                fprintf(stderr, "error: dimension cannot be negative (except -1).\n");
                return NULL;
            }
            new_size *= (uint64_t)new_shape[i];
        }
    }

    if (unknown_idx != -1 && t->size % new_size != 0) {
        fprintf(stderr, "error: invalid shape.\n");
        return NULL;
    }

    if (unknown_idx == -1 && new_size != t->size) {
        fprintf(stderr, "error: total elements must match: %" PRIu64 " != %" PRIu64 "\n", t->size, new_size);
        return NULL;
    }

    uint64_t *resolved_shape = (uint64_t *)malloc((size_t)new_ndim * sizeof(uint64_t));
    if (!resolved_shape)
        return NULL;

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
    return result;
}

/*
 * Function: tensor_transpose
 * --------------------------
 * Permutes two dimensions of the tensor.
 *
 * Example: Transpose(0, 1) on Shape (2, 3)
 * Input:
 * [[1, 2, 3],
 *  [4, 5, 6]]
 *
 * Output:
 * [[1, 4],
 *  [2, 5],
 *  [3, 6]]
 *
 * Shape: (3, 2)
 *
 * Steps:
 * 1. Validate dimensions.
 * 2. Calculate new shape (swap dim0 and dim1).
 * 3. Create result tensor.
 * 4. Iterate result indices.
 * 5. Map result indices back to input indices by swapping dims.
 * 6. Copy data element-wise.
 */
Tensor *tensor_transpose(Tensor *t, uint64_t dim0, uint64_t dim1) {
    assert(t != NULL);

    if (t->ndim < 2) {
        return tensor_create(t->data, t->shape, t->ndim, t->requires_grad);
    }

    // Bounds check
    assert(dim0 < t->ndim);
    assert(dim1 < t->ndim);

    uint64_t *new_shape = (uint64_t *)malloc((size_t)t->ndim * sizeof(uint64_t));
    if (!new_shape)
        return NULL;

    memcpy(new_shape, t->shape, (size_t)t->ndim * sizeof(uint64_t));
    uint64_t temp = new_shape[dim0];
    new_shape[dim0] = new_shape[dim1];
    new_shape[dim1] = temp;

    Tensor *result = tensor_zeros(new_shape, t->ndim, t->requires_grad);
    free(new_shape);
    if (!result)
        return NULL;

    uint64_t *indices = (uint64_t *)calloc((size_t)t->ndim, sizeof(uint64_t));
    if (!indices) {
        tensor_free(result);
        return NULL;
    }

    for (uint64_t i = 0; i < result->size; i++) {
        uint64_t temp_i = i;
        for (int64_t d = (int64_t)t->ndim - 1; d >= 0; d--) {
            indices[d] = temp_i % result->shape[d];
            temp_i /= result->shape[d];
        }

        uint64_t offset = 0;
        for (uint64_t d = 0; d < t->ndim; d++) {
            uint64_t idx_val = indices[d];
            if (d == dim0)
                idx_val = indices[dim1];
            if (d == dim1)
                idx_val = indices[dim0];

            offset += idx_val * t->strides[d];
        }

        result->data[i] = t->data[offset];
    }
    free(indices);

    assert(result != NULL);
    return result;
}

//
// reductions
//

/*
 * Function: tensor_sum
 * --------------------
 * Reduces a tensor by summing over a specified axis.
 *
 * Example: Sum axis 0 on (2, 3)
 * Input:
 * [[1, 2, 3],
 *  [4, 5, 6]]
 *
 * Summing over axis 0 (vertical):
 * [1+4, 2+5, 3+6] = [5, 7, 9]
 * Result Shape: (3) (or (1, 3) if keepdims=true)
 *
 * Steps:
 * 1. Normalize axis (handle negative indexing).
 * 2. Calculate new shape (remove axis dim unless keepdims).
 * 3. Create result tensor.
 * 4. Iterate result elements.
 * 5. For each result element:
 *    - Determine base coordinates in input tensor.
 *    - Loop over the reduction axis.
 *    - Accumulate values.
 */
// cppcheck-suppress staticFunction
Tensor *tensor_sum(Tensor *t, int64_t axis, bool keepdims) {
    assert(t != NULL);

    if (axis < 0)
        axis += (int64_t)t->ndim;

    assert(axis >= 0);
    assert(axis < (int64_t)t->ndim);

    if (axis < 0 || axis >= (int64_t)t->ndim) {
        // invalid axis
        return NULL;
    }

    uint64_t *new_shape = NULL;
    uint64_t new_ndim = keepdims ? t->ndim : t->ndim - 1;

    if (new_ndim > 0) {
        new_shape = (uint64_t *)malloc((size_t)new_ndim * sizeof(uint64_t));
        if (!new_shape)
            return NULL;
    }

    if (keepdims) {
        for (uint64_t i = 0; i < t->ndim; i++) {
            new_shape[i] = ((int64_t)i == axis) ? 1 : t->shape[i];
        }
    } else {
        uint64_t k = 0;
        for (uint64_t i = 0; i < t->ndim; i++) {
            if ((int64_t)i != axis)
                new_shape[k++] = t->shape[i];
        }
    }

    Tensor *result = tensor_zeros(new_shape, new_ndim, t->requires_grad);
    if (new_shape)
        free(new_shape);
    if (!result)
        return NULL;

    uint64_t *indices = NULL;
    if (result->ndim > 0) {
        indices = (uint64_t *)calloc((size_t)result->ndim, sizeof(uint64_t));
        if (!indices) {
            tensor_free(result);
            return NULL;
        }
    }

    for (uint64_t i = 0; i < result->size; i++) {
        if (result->ndim > 0) {
            uint64_t temp_i = i;
            for (int64_t d = (int64_t)result->ndim - 1; d >= 0; d--) {
                indices[d] = temp_i % result->shape[d];
                temp_i /= result->shape[d];
            }
        }

        uint64_t base_offset = 0;
        uint64_t k = 0;
        for (uint64_t d = 0; d < t->ndim; d++) {
            if ((int64_t)d == axis)
                continue;

            uint64_t idx_val = 0;
            if (indices) {
                idx_val = indices[keepdims ? d : k];
            }
            if (!keepdims)
                k++;

            base_offset += idx_val * t->strides[d];
        }

        float32_t sum = 0;
        uint64_t axis_dim = (t->shape) ? t->shape[axis] : 1;
        for (uint64_t j = 0; j < axis_dim; j++) {
            sum += t->data[base_offset + j * t->strides[axis]];
        }
        result->data[i] = sum;
    }

    if (indices)
        free(indices);

    assert(result != NULL);
    return result;
}

/*
 * Function: tensor_mean
 * ---------------------
 * Calculates the mean value over a specified axis.
 *
 * Steps:
 * 1. Compute sum over axis using tensor_sum.
 * 2. Divide every element in sum tensor by the size of the axis.
 */
Tensor *tensor_mean(Tensor *t, int64_t axis, bool keepdims) {
    assert(t != NULL);

    Tensor *sum = tensor_sum(t, axis, keepdims);
    if (!sum)
        return NULL;

    if (axis < 0)
        axis += (int64_t)t->ndim;

    assert(axis >= 0);
    assert(axis < (int64_t)t->ndim);

    uint64_t n = (t->shape) ? t->shape[axis] : 1;

    for (uint64_t i = 0; i < sum->size; i++) {
        sum->data[i] /= (float32_t)n;
    }

    assert(sum != NULL);
    return sum;
}

/*
 * Function: tensor_max
 * --------------------
 * Finds the maximum value over a specified axis.
 *
 * Logic is identical to tensor_sum, but accumulates Max instead of Sum.
 */
Tensor *tensor_max(Tensor *t, int64_t axis, bool keepdims) {
    assert(t != NULL);

    if (axis < 0)
        axis += (int64_t)t->ndim;

    assert(axis >= 0);
    assert(axis < (int64_t)t->ndim);

    uint64_t *new_shape = NULL;
    uint64_t new_ndim = keepdims ? t->ndim : t->ndim - 1;

    if (new_ndim > 0) {
        new_shape = (uint64_t *)malloc((size_t)new_ndim * sizeof(uint64_t));
        if (!new_shape)
            return NULL;
    }

    if (keepdims) {
        for (uint64_t i = 0; i < t->ndim; i++) {
            new_shape[i] = ((int64_t)i == axis) ? 1 : t->shape[i];
        }
    } else {
        uint64_t k = 0;
        for (uint64_t i = 0; i < t->ndim; i++) {
            if ((int64_t)i != axis)
                new_shape[k++] = t->shape[i];
        }
    }

    Tensor *result = tensor_zeros(new_shape, new_ndim, t->requires_grad);
    if (new_shape)
        free(new_shape);
    if (!result)
        return NULL;

    uint64_t *indices = NULL;
    if (result->ndim > 0) {
        indices = (uint64_t *)calloc((size_t)result->ndim, sizeof(uint64_t));
        if (!indices) {
            tensor_free(result);
            return NULL;
        }
    }

    for (uint64_t i = 0; i < result->size; i++) {
        if (indices) {
            uint64_t temp_i = i;
            for (int64_t d = (int64_t)result->ndim - 1; d >= 0; d--) {
                indices[d] = temp_i % result->shape[d];
                temp_i /= result->shape[d];
            }
        }

        uint64_t base_offset = 0;
        uint64_t k = 0;
        for (uint64_t d = 0; d < t->ndim; d++) {
            if ((int64_t)d == axis)
                continue;

            uint64_t idx_val = 0;
            if (indices) {
                idx_val = indices[keepdims ? d : k];
            }
            if (!keepdims)
                k++;

            base_offset += idx_val * t->strides[d];
        }

        float32_t max_val = -INFINITY;
        uint64_t axis_dim = (t->shape) ? t->shape[axis] : 1;
        if (axis_dim > 0) {
            max_val = t->data[base_offset];
            for (uint64_t j = 1; j < axis_dim; j++) {
                float32_t val = t->data[base_offset + j * t->strides[axis]];
                if (val > max_val)
                    max_val = val;
            }
        }
        result->data[i] = max_val;
    }

    if (indices)
        free(indices);

    assert(result != NULL);
    return result;
}

//
// utils
//

/*
 * Helper: tensor_print_recursive
 * ------------------------------
 * Recursively prints tensor data with proper formatting.
 *
 * Logic:
 * - If current dim is last dim: print elements separated by comma.
 * - Else: print opening bracket, recurse, print closing bracket.
 * - Add indentation and newlines for readability.
 */
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
            for (uint64_t j = 0; j < indent; j++)
                printf(" ");
        }
        tensor_print_recursive(t, dim + 1, offset + i * t->strides[dim], indent + 1);

        if (i < t->shape[dim] - 1) {
            printf(",");
            uint64_t newlines = t->ndim - dim - 1;
            for (uint64_t k = 0; k < newlines; k++)
                printf("\n");
        }
    }
    printf("]");
}

/*
 * Function: tensor_print
 * ----------------------
 * Prints the tensor metadata (shape, size, etc.) and data.
 * Truncates data if size > 1000.
 */
void tensor_print(Tensor *t) {
    // Assert on input, although NULL is handled by printing Tensor(NULL)
    // assert(t != NULL); // Not strictly needed as we handle it

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
        const u_int16_t max_size = 1000;
        if (t->size <= max_size) {
            printf("Data: ");
            tensor_print_recursive(t, 0, 0, 6);
            printf("\n");
        } else {
            printf("Data: ... (size > 1000)\n");
        }
    }
}

/*
 * Function: tensor_get
 * --------------------
 * Extracts a single element from the tensor as a 0-d tensor (scalar).
 *
 * Steps:
 * 1. Validate indices.
 * 2. Calculate offset = sum(index[i] * stride[i]).
 * 3. Create new 0-d tensor pointing to the data at that offset.
 */
Tensor *tensor_get(Tensor *t, const uint64_t *indices) {
    assert(t != NULL);
    assert(indices != NULL); // Assuming indices is required if ndim > 0, handled below

    uint64_t offset = 0;
    if (t->ndim > 0 && indices) {
        for (uint64_t i = 0; i < t->ndim; i++) {
            // Assert bounds? The prompt says "assert statements for every single argument".
            // Adding bounds check assertion might be good.
            assert(indices[i] < t->shape[i]);
            offset += indices[i] * t->strides[i];
        }
    }

    Tensor *res = tensor_create(&t->data[offset], NULL, 0, t->requires_grad);
    assert(res != NULL);
    return res;
}
