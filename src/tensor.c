#include "tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// helper to get size from shape
static int calculate_size(const int *shape, int ndim) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

// helper to calculate strides
static void calculate_strides(const int *shape, int ndim, int *strides) {
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

// cppcheck-suppress staticFunction
Tensor *tensor_create(const float *data, const int *shape, int ndim, bool requires_grad) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t)
        return NULL;

    t->ndim = ndim;

    // Safety check for empty shape (scalar)
    if (ndim > 0 && shape != NULL) {
        t->shape = (int *)malloc((size_t)ndim * sizeof(int));
        if (!t->shape) {
            free(t);
            return NULL;
        }
        memcpy(t->shape, shape, (size_t)ndim * sizeof(int));

        t->strides = (int *)malloc((size_t)ndim * sizeof(int));
        if (!t->strides) {
            free(t->shape);
            free(t);
            return NULL;
        }
        calculate_strides(t->shape, ndim, t->strides);
    } else {
        t->shape = NULL;
        t->strides = NULL;
        // ndim should be 0 here if consistent
        t->ndim = 0;
    }

    // Determine size. If ndim is 0, size is 1 (scalar).
    if (ndim > 0) {
        t->size = calculate_size(shape, ndim);
    } else {
        t->size = 1;
    }

    // Allocate data
    if (t->size > 0) {
        t->data = (float *)malloc((size_t)t->size * sizeof(float));
        if (!t->data) {
            if (t->strides)
                free(t->strides);
            if (t->shape)
                free(t->shape);
            free(t);
            return NULL;
        }
        if (data) {
            memcpy(t->data, data, (size_t)t->size * sizeof(float));
        } else {
            memset(t->data, 0, (size_t)t->size * sizeof(float));
        }
    } else {
        t->data = NULL;
    }

    t->requires_grad = requires_grad;
    t->grad = NULL;

    return t;
}

// cppcheck-suppress staticFunction
Tensor *tensor_zeros(const int *shape, int ndim, bool requires_grad) { return tensor_create(NULL, shape, ndim, requires_grad); }

// cppcheck-suppress staticFunction
void tensor_free(Tensor *t) {
    if (t) {
        if (t->data)
            free(t->data);
        if (t->shape)
            free(t->shape);
        if (t->strides)
            free(t->strides);
        if (t->grad)
            tensor_free(t->grad); // Recursive free if grad exists
        free(t);
    }
}

// Broadcasting logic
// Returns true if shapes can be broadcasted, and populates out_shape and out_ndim
static bool broadcast_shapes(const int *shape_a, int ndim_a, const int *shape_b, int ndim_b, int *out_shape, int *out_ndim) {
    int max_ndim = ndim_a > ndim_b ? ndim_a : ndim_b;
    *out_ndim = max_ndim;

    int idx_a = ndim_a - 1;
    int idx_b = ndim_b - 1;
    int idx_out = max_ndim - 1;

    while (idx_out >= 0) {
        int dim_a = (idx_a >= 0 && shape_a) ? shape_a[idx_a] : 1;
        int dim_b = (idx_b >= 0 && shape_b) ? shape_b[idx_b] : 1;

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

// 
// arithmetic
// 

typedef float (*binary_op_t)(float, float);

static float op_add(float a, float b) { return a + b; }
static float op_sub(float a, float b) { return a - b; }
static float op_mul(float a, float b) { return a * b; }
static float op_div(float a, float b) { return a / b; }

static Tensor *tensor_binary_op(Tensor *a, Tensor *b, binary_op_t op) {
    int out_shape[32]; // Assume max 32 dims
    int out_ndim;

    if (!broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, out_shape, &out_ndim)) {
        fprintf(stderr, "Error: Shapes cannot be broadcasted.\n");
        return NULL;
    }

    Tensor *result = tensor_zeros(out_shape, out_ndim, a->requires_grad || b->requires_grad);
    if (!result)
        return NULL;

    // Iterate over result tensor
    // Need a way to map result index to a and b indices
    int *indices = (int *)calloc((size_t)out_ndim, sizeof(int));
    if (out_ndim > 0 && !indices) { // calloc(0) depends, usually OK or NULL. Handle check carefully.
        // if out_ndim is 0, we can proceed with NULL indices?
        // if out_ndim > 0 and indices is NULL, failure.
        tensor_free(result);
        return NULL;
    }

    for (int i = 0; i < result->size; i++) {
        // Calculate indices for current element
        if (out_ndim > 0) {
            int temp = i;
            for (int d = out_ndim - 1; d >= 0; d--) {
                indices[d] = temp % out_shape[d];
                temp /= out_shape[d];
            }
        }

        // Calculate offset for a
        int offset_a = 0;
        if (a->ndim > 0) {
            for (int d = 0; d < a->ndim; d++) {
                // Broadcasting rule: align from right
                int result_dim_idx = d + (out_ndim - a->ndim);
                // If dim is 1, index is 0. Else it matches result index.
                int idx = (a->shape[d] == 1) ? 0 : indices[result_dim_idx];
                offset_a += idx * a->strides[d];
            }
        }

        // Calculate offset for b
        int offset_b = 0;
        if (b->ndim > 0) {
            for (int d = 0; d < b->ndim; d++) {
                int result_dim_idx = d + (out_ndim - b->ndim);
                int idx = (b->shape[d] == 1) ? 0 : indices[result_dim_idx];
                offset_b += idx * b->strides[d];
            }
        }

        result->data[i] = op(a->data[offset_a], b->data[offset_b]);
    }

    if (indices)
        free(indices);
    return result;
}

Tensor *tensor_add(Tensor *a, Tensor *b) { return tensor_binary_op(a, b, op_add); }
Tensor *tensor_sub(Tensor *a, Tensor *b) { return tensor_binary_op(a, b, op_sub); }
Tensor *tensor_mul(Tensor *a, Tensor *b) { return tensor_binary_op(a, b, op_mul); }
Tensor *tensor_div(Tensor *a, Tensor *b) { return tensor_binary_op(a, b, op_div); }

Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    if (a->ndim < 1 || b->ndim < 1) {
        fprintf(stderr, "Error: Matmul requires at least 1D tensors.\n");
        return NULL;
    }

    // Implementing 2D @ 2D for now as per "educational implementation" in Python
    if (a->ndim == 2 && b->ndim == 2) {
        if (a->shape[1] != b->shape[0]) {
            fprintf(stderr, "Error: Inner dimensions must match: %d != %d\n", a->shape[1], b->shape[0]);
            return NULL;
        }
        int M = a->shape[0];
        int K = a->shape[1];
        int N = b->shape[1];

        const int out_shape[] = {M, N};
        Tensor *result = tensor_zeros(out_shape, 2, a->requires_grad || b->requires_grad);
        if (!result)
            return NULL;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += a->data[i * a->strides[0] + k * a->strides[1]] * b->data[k * b->strides[0] + j * b->strides[1]];
                }
                result->data[i * result->strides[0] + j * result->strides[1]] = sum;
            }
        }
        return result;
    }

    fprintf(stderr, "Error: Only 2D matmul supported in C implementation for now.\n");
    return NULL;
}

// 
// shape manipulation
// 

Tensor *tensor_reshape(const Tensor *t, const int *new_shape, int new_ndim) {
    // Check if new shape size matches
    int new_size = 1;
    int unknown_idx = -1;
    for (int i = 0; i < new_ndim; i++) {
        if (new_shape[i] == -1) {
            if (unknown_idx != -1) {
                fprintf(stderr, "Error: Only one dimension can be -1.\n");
                return NULL;
            }
            unknown_idx = i;
        } else {
            new_size *= new_shape[i];
        }
    }

    if (unknown_idx != -1) {
        if (t->size % new_size != 0) {
            fprintf(stderr, "Error: Invalid shape.\n");
            return NULL;
        }
    } else {
        if (new_size != t->size) {
            fprintf(stderr, "Error: Total elements must match: %d != %d\n", t->size, new_size);
            return NULL;
        }
    }

    // Allocate array for resolved shape
    int *resolved_shape = (int *)malloc((size_t)new_ndim * sizeof(int));
    if (!resolved_shape)
        return NULL;

    for (int i = 0; i < new_ndim; i++) {
        resolved_shape[i] = new_shape[i];
    }
    if (unknown_idx != -1) {
        resolved_shape[unknown_idx] = t->size / new_size;
    }

    Tensor *result = tensor_create(t->data, resolved_shape, new_ndim, t->requires_grad);
    free(resolved_shape);
    return result;
}

Tensor *tensor_transpose(Tensor *t, int dim0, int dim1) {
    if (t->ndim < 2) {
        // Return copy
        return tensor_create(t->data, t->shape, t->ndim, t->requires_grad);
    }

    int *new_shape = (int *)malloc((size_t)t->ndim * sizeof(int));
    if (!new_shape)
        return NULL;
    memcpy(new_shape, t->shape, (size_t)t->ndim * sizeof(int));
    // Swap dims in shape
    int temp = new_shape[dim0];
    new_shape[dim0] = new_shape[dim1];
    new_shape[dim1] = temp;

    Tensor *result = tensor_zeros(new_shape, t->ndim, t->requires_grad);
    free(new_shape);
    if (!result)
        return NULL;

    // Transpose data
    // Iterate over result
    int *indices = (int *)calloc((size_t)t->ndim, sizeof(int));
    if (!indices) {
        tensor_free(result);
        return NULL;
    }

    for (int i = 0; i < result->size; i++) {
        // Indices in result
        int temp_i = i;
        for (int d = t->ndim - 1; d >= 0; d--) {
            indices[d] = temp_i % result->shape[d];
            temp_i /= result->shape[d];
        }

        // Map to source indices
        int offset = 0;
        for (int d = 0; d < t->ndim; d++) {
            int idx_val;
            if (d == dim0)
                idx_val = indices[dim1];
            else if (d == dim1)
                idx_val = indices[dim0];
            else
                idx_val = indices[d];

            offset += idx_val * t->strides[d];
        }

        result->data[i] = t->data[offset];
    }
    free(indices);

    return result;
}

// 
// reductions
// 

// cppcheck-suppress staticFunction
Tensor *tensor_sum(Tensor *t, int axis, bool keepdims) {
    if (axis < 0)
        axis += t->ndim;
    if (axis < 0 || axis >= t->ndim) {
        // Fallback or error
    }

    int *new_shape = NULL;
    if (t->ndim > 0) {
        new_shape = (int *)malloc((size_t)t->ndim * sizeof(int));
    }

    int new_ndim = t->ndim;

    if (keepdims) {
        if (new_shape) {
            for (int i = 0; i < t->ndim; i++) {
                new_shape[i] = (i == axis) ? 1 : t->shape[i];
            }
        }
    } else {
        new_ndim = t->ndim - 1;
        if (new_ndim > 0) {
            // Reallocate or reuse (simplification: just overwrite)
            // But we malloced size t->ndim.
            // We need to pack it.
            int k = 0;
            for (int i = 0; i < t->ndim; i++) {
                if (i != axis) {
                    new_shape[k++] = t->shape[i];
                }
            }
        }
    }

    Tensor *result = tensor_zeros(new_shape, new_ndim, t->requires_grad);
    if (new_shape)
        free(new_shape);
    if (!result)
        return NULL;

    int *indices = NULL;
    if (result->ndim > 0) {
        indices = (int *)calloc((size_t)result->ndim, sizeof(int));
        if (!indices) {
            tensor_free(result);
            return NULL;
        }
    }

    for (int i = 0; i < result->size; i++) {
        // Get current result multi-index
        if (result->ndim > 0) {
            int temp_i = i;
            for (int d = result->ndim - 1; d >= 0; d--) {
                indices[d] = temp_i % result->shape[d];
                temp_i /= result->shape[d];
            }
        }

        // Now construct base source index
        int base_offset = 0;
        int k = 0; // index in result indices
        for (int d = 0; d < t->ndim; d++) {
            if (d == axis) {
                // skip for now, will loop
            } else {
                int idx_val = 0;
                if (indices) {
                    if (keepdims) {
                        idx_val = indices[d];
                    } else {
                        idx_val = indices[k++];
                    }
                }
                base_offset += idx_val * t->strides[d];
            }
        }

        float sum = 0;
        int axis_dim = (t->shape) ? t->shape[axis] : 1; // Logic check: if ndim=0, axis=0?
        for (int j = 0; j < axis_dim; j++) {
            sum += t->data[base_offset + j * t->strides[axis]];
        }
        result->data[i] = sum;
    }

    if (indices)
        free(indices);
    return result;
}

Tensor *tensor_mean(Tensor *t, int axis, bool keepdims) {
    Tensor *sum = tensor_sum(t, axis, keepdims);
    if (!sum)
        return NULL;

    if (axis < 0)
        axis += t->ndim;
    int n = (t->shape) ? t->shape[axis] : 1;

    // Div by n
    for (int i = 0; i < sum->size; i++) {
        sum->data[i] /= (float)n;
    }

    return sum;
}

Tensor *tensor_max(Tensor *t, int axis, bool keepdims) {
    if (axis < 0)
        axis += t->ndim;

    int *new_shape = NULL;
    if (t->ndim > 0) {
        new_shape = (int *)malloc((size_t)t->ndim * sizeof(int));
    }
    int new_ndim = t->ndim;

    if (keepdims) {
        if (new_shape) {
            for (int i = 0; i < t->ndim; i++) {
                new_shape[i] = (i == axis) ? 1 : t->shape[i];
            }
        }
    } else {
        new_ndim = t->ndim - 1;
        if (new_ndim > 0) {
            int k = 0;
            for (int i = 0; i < t->ndim; i++) {
                if (i != axis) {
                    new_shape[k++] = t->shape[i];
                }
            }
        }
    }

    Tensor *result = tensor_zeros(new_shape, new_ndim, t->requires_grad);
    if (new_shape)
        free(new_shape);
    if (!result)
        return NULL;

    int *indices = NULL;
    if (result->ndim > 0) {
        indices = (int *)calloc((size_t)result->ndim, sizeof(int));
    }

    for (int i = 0; i < result->size; i++) {
        if (indices) {
            int temp_i = i;
            for (int d = result->ndim - 1; d >= 0; d--) {
                indices[d] = temp_i % result->shape[d];
                temp_i /= result->shape[d];
            }
        }

        int base_offset = 0;
        int k = 0;
        for (int d = 0; d < t->ndim; d++) {
            if (d == axis) {
            } else {
                int idx_val = 0;
                if (indices) {
                    if (keepdims) {
                        idx_val = indices[d];
                    } else {
                        idx_val = indices[k++];
                    }
                }
                base_offset += idx_val * t->strides[d];
            }
        }

        float max_val = -INFINITY;
        int axis_dim = (t->shape) ? t->shape[axis] : 1;
        if (axis_dim > 0) {
            max_val = t->data[base_offset]; // Start with first
            for (int j = 1; j < axis_dim; j++) {
                float val = t->data[base_offset + j * t->strides[axis]];
                if (val > max_val)
                    max_val = val;
            }
        }
        result->data[i] = max_val;
    }

    if (indices)
        free(indices);
    return result;
}

// 
// utils
// 

void tensor_print(Tensor *t) {
    if (!t) {
        printf("Tensor(NULL)\n");
        return;
    }
    printf("Tensor(shape=[");
    if (t->shape) {
        for (int i = 0; i < t->ndim; i++) {
            printf("%d%s", t->shape[i], i < t->ndim - 1 ? ", " : "");
        }
    }
    printf("], size=%d, requires_grad=%s)\n", t->size, t->requires_grad ? "true" : "false");
    // TODO: print data nicely for higher dimensions
    if (t->data && t->size <= 20) {
        printf("Data: [");
        for (int i = 0; i < t->size; i++) {
            printf("%f%s", t->data[i], i < t->size - 1 ? ", " : "");
        }
        printf("]\n");
    }
}

Tensor *tensor_get(Tensor *t, const int *indices) {
    int offset = 0;
    if (t->ndim > 0 && indices) {
        for (int i = 0; i < t->ndim; i++) {
            offset += indices[i] * t->strides[i];
        }
    }

    Tensor *res = tensor_create(&t->data[offset], NULL, 0, t->requires_grad); // shape NULL, ndim 0 -> scalar
    return res;
}
