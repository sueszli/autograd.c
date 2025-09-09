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
#include "../utils/types.h"
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

// helper function to calculate linear index from multi-dimensional indices
static u64 calculate_index(const i32 *indices, const i32 *shape, i32 ndim) {
    u64 index = 0;
    u64 stride = 1;

    for (i32 i = ndim - 1; i >= 0; i--) {
        index += (u64)indices[i] * stride;
        stride *= (u64)shape[i];
    }

    return index;
}

// helper function to convert linear index to multi-dimensional indices
static void linear_to_indices(u64 linear_index, const i32 *shape, i32 ndim, i32 *indices) {
    for (i32 i = ndim - 1; i >= 0; i--) {
        indices[i] = (i32)(linear_index % (u64)shape[i]);
        linear_index /= (u64)shape[i];
    }
}

// Broadcast a tensor to a new shape
tensor_t *tensor_broadcast_to(const tensor_t *tensor, const i32 *target_shape, i32 target_ndim) {
    if (!tensor || !target_shape)
        return NULL;

    // Check if broadcasting is possible
    for (i32 i = 0; i < target_ndim; i++) {
        i32 tensor_dim = (i < tensor->ndim) ? tensor->shape[tensor->ndim - 1 - i] : 1;
        i32 target_dim = target_shape[target_ndim - 1 - i];

        if (tensor_dim != target_dim && tensor_dim != 1) {
            return NULL;
        }
    }

    // Calculate total size of target tensor
    u64 target_size = 1;
    for (i32 i = 0; i < target_ndim; i++) {
        target_size *= (u64)target_shape[i];
    }

    // Create result tensor
    i32 *shape_copy = (i32 *)malloc((u64)target_ndim * sizeof(i32));
    for (i32 i = 0; i < target_ndim; i++) {
        shape_copy[i] = target_shape[i];
    }
    tensor_t *result = tensor_create(NULL, shape_copy, target_ndim, tensor->requires_grad);
    free(shape_copy); // Free the temporary copy since tensor_create makes its own copy

    // Fill the broadcasted data
    i32 *target_indices = (i32 *)malloc((u64)target_ndim * sizeof(i32));
    i32 *source_indices = (i32 *)malloc((u64)tensor->ndim * sizeof(i32));

    for (u64 i = 0; i < target_size; i++) {
        // Convert linear index to multi-dimensional indices for target
        linear_to_indices(i, target_shape, target_ndim, target_indices);

        // Map target indices to source indices
        for (i32 j = 0; j < tensor->ndim; j++) {
            i32 target_idx = target_ndim - tensor->ndim + j;
            if (target_idx >= 0) {
                source_indices[j] = tensor->shape[j] == 1 ? 0 : target_indices[target_idx];
            } else {
                source_indices[j] = 0;
            }
        }

        // Calculate source index and copy data
        u64 source_idx = calculate_index(source_indices, tensor->shape, tensor->ndim);
        result->data[i] = tensor->data[source_idx];
    }

    free(target_indices);
    free(source_indices);

    return result;
}
