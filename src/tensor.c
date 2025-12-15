#include "tensor.h"
#include "ops/arithmetic.h"
#include "utils/aligned_alloc.h"
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//
// memory management
//

static uint64_t size(const uint64_t *shape, uint64_t ndim) {
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
static uint64_t *strides(const uint64_t *shape, uint64_t ndim) {
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

    t->strides = strides(t->shape, ndim);

    t->size = size(shape, ndim);

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
    assert(t->size == size(shape, ndim));
    assert(t->data != NULL || t->size == 0);
    return t;
}

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
void linear_to_multidim_mut(uint64_t lin, const uint64_t *shape, uint64_t ndim, uint64_t *out_multidim) {
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
 * converts multi-dimensional coordinates to a linear memory offset.
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
uint64_t multidim_to_linear(const uint64_t *target, uint64_t target_ndim, const uint64_t *shape, uint64_t ndim, const uint64_t *strides) {
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

//
// utils
//

static void tensor_print_recursive(const Tensor *t, uint64_t dim, uint64_t offset, uint64_t indent) {
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

void tensor_print(const Tensor *t) {
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
Tensor *tensor_get(const Tensor *t, const uint64_t *multidim) {
    if (!t) {
        return NULL;
    }
    assert(multidim != NULL);
    assert(t->data != NULL);
    assert(t->ndim <= MAX_NDIM);

    uint64_t offset = multidim_to_linear(multidim, t->ndim, t->shape, t->ndim, t->strides);
    assert(offset < t->size);

    // scalar tensor with 0 dim
    Tensor *val = tensor_create(&t->data[offset], NULL, 0, t->requires_grad);
    return val;
}
