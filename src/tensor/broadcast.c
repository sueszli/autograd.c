#include "broadcast.h"
#include "../utils/types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool tensor_can_broadcast(const Tensor *a, const Tensor *b) {
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

// get the resulting shape after broadcasting two tensors
i32 *get_tensor_broadcast_shape(const Tensor *a, const Tensor *b, i32 *result_ndim) {
    if (!tensor_can_broadcast(a, b)) {
        *result_ndim = 0;
        return NULL;
    }

    *result_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    i32 *result_shape = (i32 *)malloc((size_t)*result_ndim * sizeof(i32));

    for (i32 i = 0; i < *result_ndim; i++) {
        i32 dim_a = (i < a->ndim) ? a->shape[a->ndim - 1 - i] : 1;
        i32 dim_b = (i < b->ndim) ? b->shape[b->ndim - 1 - i] : 1;

        result_shape[*result_ndim - 1 - i] = dim_a > dim_b ? dim_a : dim_b;
    }

    return result_shape;
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
Tensor *tensor_broadcast_to(const Tensor *tensor, const i32 *target_shape, i32 target_ndim) {
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
    i32 *shape_copy = (i32 *)malloc((size_t)target_ndim * sizeof(i32));
    for (i32 i = 0; i < target_ndim; i++) {
        shape_copy[i] = target_shape[i];
    }
    Tensor *result = tensor_create(NULL, shape_copy, target_ndim, tensor->requires_grad);
    free(shape_copy); // Free the temporary copy since tensor_create makes its own copy

    // Fill the broadcasted data
    i32 *target_indices = (i32 *)malloc((size_t)target_ndim * sizeof(i32));
    i32 *source_indices = (i32 *)malloc((size_t)tensor->ndim * sizeof(i32));

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

// Add two tensors with broadcasting support
Tensor *tensor_add_broadcast(Tensor *a, Tensor *b) {
    if (!tensor_can_broadcast(a, b)) {
        return NULL;
    }

    // If shapes are identical, use regular addition
    if (a->ndim == b->ndim) {
        bool same_shape = true;
        for (i32 i = 0; i < a->ndim; i++) {
            if (a->shape[i] != b->shape[i]) {
                same_shape = false;
                break;
            }
        }
        if (same_shape) {
            return tensor_add(a, b);
        }
    }

    // Get broadcast shape
    i32 result_ndim;
    i32 *result_shape = get_tensor_broadcast_shape(a, b, &result_ndim);
    if (!result_shape)
        return NULL;

    // Broadcast both tensors to the result shape
    Tensor *a_broadcast = tensor_broadcast_to(a, result_shape, result_ndim);
    Tensor *b_broadcast = tensor_broadcast_to(b, result_shape, result_ndim);

    if (!a_broadcast || !b_broadcast) {
        free(result_shape);
        if (a_broadcast)
            tensor_destroy(a_broadcast);
        if (b_broadcast)
            tensor_destroy(b_broadcast);
        return NULL;
    }

    // Perform addition on broadcasted tensors
    Tensor *result = tensor_add(a_broadcast, b_broadcast);

    // Clean up
    tensor_destroy(a_broadcast);
    tensor_destroy(b_broadcast);
    free(result_shape);

    return result;
}

// Multiply two tensors with broadcasting support
Tensor *tensor_mul_broadcast(Tensor *a, Tensor *b) {
    if (!tensor_can_broadcast(a, b)) {
        return NULL;
    }

    // If shapes are identical, use regular multiplication
    if (a->ndim == b->ndim) {
        bool same_shape = true;
        for (i32 i = 0; i < a->ndim; i++) {
            if (a->shape[i] != b->shape[i]) {
                same_shape = false;
                break;
            }
        }
        if (same_shape) {
            return tensor_mul(a, b);
        }
    }

    // Get broadcast shape
    i32 result_ndim;
    i32 *result_shape = get_tensor_broadcast_shape(a, b, &result_ndim);
    if (!result_shape)
        return NULL;

    // Broadcast both tensors to the result shape
    Tensor *a_broadcast = tensor_broadcast_to(a, result_shape, result_ndim);
    Tensor *b_broadcast = tensor_broadcast_to(b, result_shape, result_ndim);

    if (!a_broadcast || !b_broadcast) {
        free(result_shape);
        if (a_broadcast)
            tensor_destroy(a_broadcast);
        if (b_broadcast)
            tensor_destroy(b_broadcast);
        return NULL;
    }

    // Perform multiplication on broadcasted tensors
    Tensor *result = tensor_mul(a_broadcast, b_broadcast);

    // Clean up
    tensor_destroy(a_broadcast);
    tensor_destroy(b_broadcast);
    free(result_shape);

    return result;
}