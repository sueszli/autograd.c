//
// broadcasting means expanding the smaller tensor to match the larger one.
// we follow the numpy/pytorch broadcasting convention.
//
// example: adding a 2x1 tensor to a 1x3 tensor results in a 2x3 tensor
//
//    A (2x1)       B (1x3)      Result (2x3)
//    ┌─────┐    ┌──────────┐    ┌──────────┐
//    │  1  │ +  │ 10 20 30 │ =  │ 11 21 31 │
//    │  4  │    └──────────┘    │ 14 24 34 │
//    └─────┘                    └──────────┘
//
// steps:
//                       ┌─────┐   ┌───────────┐
//    (1) A is expanded: │  1  │ → │ 1   1   1 │ (repeat across columns)
//                       │  4  │   │ 4   4   4 │
//                       └─────┘   └───────────┘
//
//                       ┌──────────┐   ┌──────────┐
//    (2) B is expanded: │ 10 20 30 │ → │ 10 20 30 │ (repeat across rows)
//                       └──────────┘   │ 10 20 30 │
//                                      └──────────┘
//
//    (3) operation is performed element-wise
//
#include "broadcast.h"
#include "../utils/defer.h"
#include "../utils/types.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void shape_free(shape_t *s) {
    if (s && s->shape) {
        free(s->shape);
        s->shape = NULL;
        s->ndim = 0;
    }
}

// rules for broadcasting:
//
// - the trailing dimensions must match or one must be 1.
// - leading dimensions can be missing (treated as 1).
//
// we just look at the shape of the tensors, not the data.
//
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
bool tensor_can_broadcast(const tensor_t *a, const tensor_t *b) {
    if (!a || !b) {
        return false;
    }

    i32 max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;

    for (i32 i = 0; i < max_ndim; i++) {
        i32 dim_a = (i < a->ndim) ? a->shape[a->ndim - 1 - i] : 1;
        i32 dim_b = (i < b->ndim) ? b->shape[b->ndim - 1 - i] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }

    return true;
}

bool tensor_can_broadcast_to_shape(const tensor_t *tensor, const i32 *target_shape, i32 target_ndim) {
    if (!tensor || !target_shape) {
        return false;
    }

    for (i32 i = 0; i < target_ndim; i++) {
        i32 tensor_dim = (i < tensor->ndim) ? tensor->shape[tensor->ndim - 1 - i] : 1;
        i32 target_dim = target_shape[target_ndim - 1 - i];

        if (tensor_dim != target_dim && tensor_dim != 1) {
            return false;
        }
    }

    return true;
}

shape_t get_tensor_broadcast_shape(const tensor_t *a, const tensor_t *b) {
    shape_t result = {NULL, 0};

    if (!tensor_can_broadcast(a, b)) {
        return result;
    }

    result.ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    result.shape = (i32 *)malloc((u64)result.ndim * sizeof(i32));

    for (i32 i = 0; i < result.ndim; i++) {
        i32 dim_a = (i < a->ndim) ? a->shape[a->ndim - 1 - i] : 1;
        i32 dim_b = (i < b->ndim) ? b->shape[b->ndim - 1 - i] : 1;

        result.shape[result.ndim - 1 - i] = dim_a > dim_b ? dim_a : dim_b;
    }

    return result;
}

// helper function to convert multi-dimensional indices to linear index
//
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
    u64 index = 0;
    u64 stride = 1;

    for (i32 i = ndim - 1; i >= 0; i--) {
        index += (u64)indices[i] * stride;
        stride *= (u64)shape[i];
    }

    return index;
}

// helper function to convert linear index to multi-dimensional indices
static i32 *get_multi_dim_idx(u64 linear_index, const i32 *shape, i32 ndim) {
    i32 *indices = (i32 *)malloc((u64)ndim * sizeof(i32));
    assert(indices != NULL);

    for (i32 i = ndim - 1; i >= 0; i--) {
        indices[i] = (i32)(linear_index % (u64)shape[i]);
        linear_index /= (u64)shape[i];
    }
    return indices;
}

// broadcast a tensor to a new shape
//
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
tensor_t *tensor_broadcast_to(const tensor_t *tensor, const i32 *target_shape, i32 target_ndim) {
    if (!tensor_can_broadcast_to_shape(tensor, target_shape, target_ndim)) {
        return NULL;
    }

    // malloc target tensor (to write into)
    i32 *shape_copy = (i32 *)malloc((u64)target_ndim * sizeof(i32));
    for (i32 i = 0; i < target_ndim; i++) {
        shape_copy[i] = target_shape[i];
    }
    tensor_t *target = tensor_create(NULL, shape_copy, target_ndim, tensor->requires_grad);
    free(shape_copy);

    // malloc source index array
    i32 *source_indices = (i32 *)malloc((u64)tensor->ndim * sizeof(i32));
    defer({ free(source_indices); });

    // copy data from source to target
    // by iterating in linear index space of target
    u64 target_size = 1;
    for (i32 i = 0; i < target_ndim; i++) {
        target_size *= (u64)target_shape[i];
    }
    for (u64 i = 0; i < target_size; i++) {

        // get target coordinates
        i32 *target_indices = get_multi_dim_idx(i, target_shape, target_ndim);
        assert(target_indices != NULL);

        // get source coordinates (expressed as an array of indices)
        for (i32 j = 0; j < tensor->ndim; j++) {
            i32 target_idx = target_ndim - tensor->ndim + j;
            if (target_idx >= 0) {
                source_indices[j] = tensor->shape[j] == 1 ? 0 : target_indices[target_idx];
            } else {
                source_indices[j] = 0;
            }
        }

        // copy
        u64 source_idx = get_linear_idx(source_indices, tensor->shape, tensor->ndim);
        target->data[i] = tensor->data[source_idx];

        free(target_indices);
    }

    return target;
}

bool tensor_shapes_match(const tensor_t *a, const tensor_t *b) {
    if (!a || !b) {
        return false;
    }

    if (a->ndim != b->ndim) {
        return false;
    }

    for (i32 i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }

    return true;
}

void tensor_broadcast_inplace(tensor_t **a, tensor_t **b) {
    if (!a || !*a || !b || !*b) {
        return;
    }

    if (!tensor_can_broadcast(*a, *b)) {
        return;
    }

    if (tensor_shapes_match(*a, *b)) {
        return;
    }

    shape_t broadcast_shape = get_tensor_broadcast_shape(*a, *b);
    if (!broadcast_shape.shape) {
        return;
    }
    defer({ shape_free(&broadcast_shape); });

    tensor_t *new_a = tensor_broadcast_to(*a, broadcast_shape.shape, broadcast_shape.ndim);
    tensor_t *new_b = tensor_broadcast_to(*b, broadcast_shape.shape, broadcast_shape.ndim);

    if (new_a && new_b) {
        tensor_destroy(*a);
        tensor_destroy(*b);
        *a = new_a;
        *b = new_b;
    } else {
        if (new_a) tensor_destroy(new_a);
        if (new_b) tensor_destroy(new_b);
    }
}
