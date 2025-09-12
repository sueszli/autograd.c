#include "broadcast.h"
#include "../utils/defer.h"
#include "../utils/types.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline i32 imax(i32 a, i32 b) { return (a > b) ? a : b; }

static inline i32 get_dim(const tensor_t *tensor, i32 i) {
    assert(tensor != NULL);
    return (i < tensor->ndim) ? tensor->shape[tensor->ndim - 1 - i] : 1;
}

// example: [3,1,4] and [2,4]
//
// position:  2  1  0  ← dimension positions (right-to-left)
//            ↑  ↑  ↑
// tensor A: [3, 1, 4]
// tensor B:    [2, 4]  ← B doesn't have position 2
//
// (1) compare: A[0] = 4 vs B[0] = 4 ✓
// (2) compare: A[1] = 1 vs B[1] = 2 ✓ (1 can broadcast)
// (3) compare: A[2] = 3 vs B[2] = (missing, treated as 1) ✓
// (4) result shape: [3, 2, 4]
//
static bool tensor_can_broadcast(const tensor_t *a, const tensor_t *b) {
    assert(a != NULL && b != NULL);
    i32 max_ndim = imax(a->ndim, b->ndim);
    for (i32 i = 0; i < max_ndim; i++) {
        i32 dim_a = get_dim(a, i);
        i32 dim_b = get_dim(b, i);
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }
    return true;
}

static bool tensor_can_broadcast_to_shape(const tensor_t *tensor, const i32 *target_shape, i32 target_ndim) {
    assert(tensor != NULL && target_shape != NULL);
    for (i32 i = 0; i < target_ndim; i++) {
        i32 tensor_dim = get_dim(tensor, i);
        i32 target_dim = target_shape[target_ndim - 1 - i];
        if (tensor_dim != target_dim && tensor_dim != 1) {
            return false;
        }
    }
    return true;
}

static i32 *get_broadcast_shape(const tensor_t *a, const tensor_t *b, i32 *out_ndim) {
    assert(a != NULL && b != NULL && out_ndim != NULL);
    assert(tensor_can_broadcast(a, b));

    *out_ndim = imax(a->ndim, b->ndim);
    i32 *shape = (i32 *)malloc((u64)*out_ndim * sizeof(i32));
    assert(shape != NULL);

    for (i32 i = 0; i < *out_ndim; i++) {
        i32 dim_a = get_dim(a, i);
        i32 dim_b = get_dim(b, i);
        shape[*out_ndim - 1 - i] = imax(dim_a, dim_b);
    }

    return shape;
}

// example:
//
// ┌─────┬─────┬─────┐
// │ [0] │ [1] │ [2] │
// ├─────┼─────┼─────┤
// │ [3] │ [4] │ [5] │
// └─────┴─────┴─────┘
//
// - indices = [1, 2]    (argument to convert into linear index)
// - shape = [2, 3]      (2 rows, 3 columns)
// - ndim = 2            (length of shape)
//
// result: 1*3 + 2 = 5
//
static u64 get_linear_idx(const i32 *indices, const i32 *shape, i32 ndim) {
    assert(indices != NULL && shape != NULL);
    u64 index = 0;
    u64 stride = 1;
    for (i32 i = ndim - 1; i >= 0; i--) {
        index += (u64)indices[i] * stride;
        stride *= (u64)shape[i];
    }
    return index;
}

static i32 *get_multi_dim_idx(u64 linear_index, const i32 *shape, i32 ndim) {
    assert(shape != NULL);
    i32 *indices = (i32 *)malloc((u64)ndim * sizeof(i32));
    assert(indices != NULL);
    for (i32 i = ndim - 1; i >= 0; i--) {
        indices[i] = (i32)(linear_index % (u64)shape[i]);
        linear_index /= (u64)shape[i];
    }
    return indices;
}

// example:
//
//      source tensor (2x1):        target shape (2x3):
//      ┌───┐                       ┌───────────┐
//      │ 5 │    - broadcast to →   │ 5   5   5 │
//      │ 7 │                       │ 7   7   7 │
//      └───┘                       └───────────┘
//
// iteration | target [row,col] | source [row,col] | source linear | value copied
// ----------|------------------|------------------|---------------|-------------
//     0     |     [0,0]        |     [0,0]        |       0       |      5
//     1     |     [0,1]        |     [0,0]        |       0       |      5
//     2     |     [0,2]        |     [0,0]        |       0       |      5
//     3     |     [1,0]        |     [1,0]        |       1       |      7
//     4     |     [1,1]        |     [1,0]        |       1       |      7
//     5     |     [1,2]        |     [1,0]        |       1       |      7
//
static tensor_t *tensor_broadcast_to(const tensor_t *tensor, const i32 *target_shape, i32 target_ndim) {
    assert(tensor != NULL && target_shape != NULL);
    assert(tensor_can_broadcast_to_shape(tensor, target_shape, target_ndim));

    i32 *shape_copy = (i32 *)malloc((u64)target_ndim * sizeof(i32));
    assert(shape_copy != NULL);
    memcpy(shape_copy, target_shape, (u64)target_ndim * sizeof(i32));

    tensor_t *target = tensor_create(NULL, shape_copy, target_ndim, tensor->requires_grad);
    free(shape_copy);
    assert(target != NULL);

    i32 *source_indices = (i32 *)malloc((u64)tensor->ndim * sizeof(i32));
    assert(source_indices != NULL);
    defer({ free(source_indices); });

    u64 target_size = tensor_size(target);
    for (u64 i = 0; i < target_size; i++) {
        i32 *target_indices = get_multi_dim_idx(i, target_shape, target_ndim);
        defer({ free(target_indices); });

        for (i32 j = 0; j < tensor->ndim; j++) {
            i32 target_idx = target_ndim - tensor->ndim + j;
            if (target_idx >= 0) {
                source_indices[j] = tensor->shape[j] == 1 ? 0 : target_indices[target_idx];
            } else {
                source_indices[j] = 0;
            }
        }

        u64 source_idx = get_linear_idx(source_indices, tensor->shape, tensor->ndim);
        target->data[i] = tensor->data[source_idx];
    }

    return target;
}

bool tensor_shapes_match(const tensor_t *a, const tensor_t *b) {
    assert(a != NULL && b != NULL);
    if (a->ndim != b->ndim)
        return false;
    for (i32 i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i])
            return false;
    }
    return true;
}

broadcasted_tensors_t tensor_broadcast(tensor_t *a, tensor_t *b) {
    assert(a != NULL && b != NULL);

    if (tensor_shapes_match(a, b)) {
        return (broadcasted_tensors_t){a, b};
    }

    assert(tensor_can_broadcast(a, b));

    i32 broadcast_ndim;
    i32 *broadcast_shape = get_broadcast_shape(a, b, &broadcast_ndim);
    defer({ free(broadcast_shape); });

    tensor_t *b_a = tensor_broadcast_to(a, broadcast_shape, broadcast_ndim);
    tensor_t *b_b = tensor_broadcast_to(b, broadcast_shape, broadcast_ndim);

    return (broadcasted_tensors_t){b_a, b_b};
}

tensor_t *tensor_reduce(const tensor_t *broadcasted_grad, const tensor_t *target_tensor) {
    assert(broadcasted_grad != NULL && target_tensor != NULL);

    if (tensor_shapes_match(broadcasted_grad, target_tensor)) {
        i32 *shape_copy = (i32 *)malloc((u64)target_tensor->ndim * sizeof(i32));
        assert(shape_copy != NULL);
        memcpy(shape_copy, target_tensor->shape, (u64)target_tensor->ndim * sizeof(i32));

        tensor_t *result = tensor_create(NULL, shape_copy, target_tensor->ndim, false);
        free(shape_copy);

        u64 size = tensor_size(target_tensor);
        memcpy(result->data, broadcasted_grad->data, size * sizeof(f32));
        return result;
    }

    u64 target_size = tensor_size(target_tensor);
    u64 broadcasted_size = tensor_size(broadcasted_grad);

    i32 *target_shape_copy = (i32 *)malloc((u64)target_tensor->ndim * sizeof(i32));
    assert(target_shape_copy != NULL);
    memcpy(target_shape_copy, target_tensor->shape, (u64)target_tensor->ndim * sizeof(i32));

    tensor_t *result = tensor_create(NULL, target_shape_copy, target_tensor->ndim, false);
    free(target_shape_copy);

    for (u64 i = 0; i < target_size; i++) {
        result->data[i] = 0.0f;
    }

    i32 *broadcasted_indices = (i32 *)malloc((u64)broadcasted_grad->ndim * sizeof(i32));
    i32 *target_indices = (i32 *)malloc((u64)target_tensor->ndim * sizeof(i32));
    assert(broadcasted_indices != NULL && target_indices != NULL);
    defer({
        free(broadcasted_indices);
        free(target_indices);
    });

    for (u64 idx = 0; idx < broadcasted_size; idx++) {
        i32 *multi_dim_idx = get_multi_dim_idx(idx, broadcasted_grad->shape, broadcasted_grad->ndim);
        defer({ free(multi_dim_idx); });

        for (i32 i = 0; i < target_tensor->ndim; i++) {
            i32 broadcast_dim_idx = broadcasted_grad->ndim - target_tensor->ndim + i;
            if (broadcast_dim_idx >= 0) {
                target_indices[i] = target_tensor->shape[i] == 1 ? 0 : multi_dim_idx[broadcast_dim_idx];
            } else {
                target_indices[i] = 0;
            }
        }

        u64 target_idx = get_linear_idx(target_indices, target_tensor->shape, target_tensor->ndim);
        result->data[target_idx] += broadcasted_grad->data[idx];
    }

    return result;
}