#include "tensor.h"
#include "../utils/types.h"
#include "autograd.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to calculate the total number of elements in a tensor
u64 tensor_size(const tensor_t *t) {
    if (t == NULL)
        return 0;
    u64 size = 1;
    for (i32 i = 0; i < t->ndim; i++) {
        size *= (u64)t->shape[i];
    }
    return size;
}

// Create a new tensor
tensor_t *tensor_create(f32 *data, i32 *shape, i32 ndim, bool requires_grad) {
    tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
    t->shape = (i32 *)malloc((u64)ndim * sizeof(i32));
    memcpy(t->shape, shape, (u64)ndim * sizeof(i32));
    t->ndim = ndim;

    u64 size = 1;
    for (i32 i = 0; i < ndim; i++) {
        size *= (u64)shape[i];
    }

    t->data = (f32 *)malloc(size * sizeof(f32));
    if (data != NULL) {
        memcpy(t->data, data, size * sizeof(f32));
    }

    t->requires_grad = requires_grad;
    t->grad = NULL;
    t->grad_fn = NULL;
    t->ctx = NULL;
    t->ctx_size = 0;

    return t;
}

// Destroy a tensor and its data
void tensor_destroy(tensor_t *t) {
    if (t == NULL)
        return;
    free(t->data);
    free(t->shape);
    if (t->grad) {
        tensor_destroy(t->grad);
    }
    if (t->ctx) {
        free(t->ctx);
    }
    free(t);
}

// print a tensor (simple implementation)
void tensor_print(const tensor_t *t) {
    printf("tensor_t (shape: [");
    for (i32 i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if (i < t->ndim - 1)
            printf(", ");
    }
    printf("], requires_grad: %s)\n", t->requires_grad ? "true" : "false");

    u64 size = tensor_size(t);
    for (u64 i = 0; i < size; i++) {
        printf("%f ", t->data[i]);
    }
    printf("\n");
}

void tensor_zero_grad(tensor_t *t) {
    if (t->grad) {
        memset(t->grad->data, 0, tensor_size(t->grad) * sizeof(f32));
    }
}

// Backward function for addition
void add_backward(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];

    if (a->requires_grad) {
        for (u64 i = 0; i < tensor_size(a); i++) {
            if (a->grad == NULL) {
                a->grad = tensor_create(NULL, a->shape, a->ndim, false);
                memset(a->grad->data, 0, tensor_size(a) * sizeof(f32));
            }
            a->grad->data[i] += t->grad->data[i];
        }
    }
    if (b->requires_grad) {
        for (u64 i = 0; i < tensor_size(b); i++) {
            if (b->grad == NULL) {
                b->grad = tensor_create(NULL, b->shape, b->ndim, false);
                memset(b->grad->data, 0, tensor_size(b) * sizeof(f32));
            }
            b->grad->data[i] += t->grad->data[i];
        }
    }
}

// tensor_t addition
tensor_t *tensor_add(tensor_t *a, tensor_t *b) {
    // Basic shape check
    if (a->ndim != b->ndim)
        return NULL;
    for (i32 i = 0; i < a->ndim; ++i) {
        if (a->shape[i] != b->shape[i])
            return NULL;
    }

    f32 *new_data = (f32 *)malloc(tensor_size(a) * sizeof(f32));
    for (u64 i = 0; i < tensor_size(a); i++) {
        new_data[i] = a->data[i] + b->data[i];
    }

    bool requires_grad = a->requires_grad || b->requires_grad;
    tensor_t *result = tensor_create(new_data, a->shape, a->ndim, requires_grad);
    free(new_data);

    if (requires_grad) {
        result->grad_fn = add_backward;
        result->ctx = (void **)malloc(2 * sizeof(void *));
        result->ctx[0] = a;
        result->ctx[1] = b;
        result->ctx_size = 2;
    }

    return result;
}

// Backward function for multiplication
void mul_backward(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];

    if (a->requires_grad) {
        for (u64 i = 0; i < tensor_size(a); i++) {
            if (a->grad == NULL) {
                a->grad = tensor_create(NULL, a->shape, a->ndim, false);
                memset(a->grad->data, 0, tensor_size(a) * sizeof(f32));
            }
            a->grad->data[i] += b->data[i] * t->grad->data[i];
        }
    }
    if (b->requires_grad) {
        for (u64 i = 0; i < tensor_size(b); i++) {
            if (b->grad == NULL) {
                b->grad = tensor_create(NULL, b->shape, b->ndim, false);
                memset(b->grad->data, 0, tensor_size(b) * sizeof(f32));
            }
            b->grad->data[i] += a->data[i] * t->grad->data[i];
        }
    }
}

// tensor_t multiplication
tensor_t *tensor_mul(tensor_t *a, tensor_t *b) {
    // Basic shape check
    if (a->ndim != b->ndim)
        return NULL;
    for (i32 i = 0; i < a->ndim; ++i) {
        if (a->shape[i] != b->shape[i])
            return NULL;
    }

    f32 *new_data = (f32 *)malloc(tensor_size(a) * sizeof(f32));
    for (u64 i = 0; i < tensor_size(a); i++) {
        new_data[i] = a->data[i] * b->data[i];
    }

    bool requires_grad = a->requires_grad || b->requires_grad;
    tensor_t *result = tensor_create(new_data, a->shape, a->ndim, requires_grad);
    free(new_data);

    if (requires_grad) {
        result->grad_fn = mul_backward;
        result->ctx = (void **)malloc(2 * sizeof(void *));
        result->ctx[0] = a;
        result->ctx[1] = b;
        result->ctx_size = 2;
    }

    return result;
}

// Backward function for relu
void relu_backward(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    if (a->requires_grad) {
        for (u64 i = 0; i < tensor_size(a); i++) {
            if (a->grad == NULL) {
                a->grad = tensor_create(NULL, a->shape, a->ndim, false);
                memset(a->grad->data, 0, tensor_size(a) * sizeof(f32));
            }
            a->grad->data[i] += (a->data[i] > 0) * t->grad->data[i];
        }
    }
}

// ReLU activation
tensor_t *tensor_relu(tensor_t *a) {
    f32 *new_data = (f32 *)malloc(tensor_size(a) * sizeof(f32));
    for (u64 i = 0; i < tensor_size(a); i++) {
        new_data[i] = a->data[i] > 0 ? a->data[i] : 0;
    }

    tensor_t *result = tensor_create(new_data, a->shape, a->ndim, a->requires_grad);
    free(new_data);

    if (a->requires_grad) {
        result->grad_fn = relu_backward;
        result->ctx = (void **)malloc(sizeof(void *));
        result->ctx[0] = a;
        result->ctx_size = 1;
    }

    return result;
}

tensor_t *tensor_transpose(tensor_t *a) {
    if (a->ndim != 2)
        return NULL;
    i32 new_shape[] = {a->shape[1], a->shape[0]};
    f32 *new_data = malloc(tensor_size(a) * sizeof(f32));

    for (i32 i = 0; i < a->shape[0]; i++) {
        for (i32 j = 0; j < a->shape[1]; j++) {
            new_data[j * a->shape[0] + i] = a->data[i * a->shape[1] + j];
        }
    }

    tensor_t *result = tensor_create(new_data, new_shape, 2, a->requires_grad);
    free(new_data);
    result->requires_grad = false; // Transpose backward is not implemented for simplicity

    return result;
}

// Backward function for matmul
void matmul_backward(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];

    if (a->requires_grad) {
        tensor_t *b_t = tensor_transpose(b);
        tensor_t *da = tensor_matmul(t->grad, b_t);
        if (a->grad == NULL) {
            a->grad = tensor_create(NULL, a->shape, a->ndim, false);
            memset(a->grad->data, 0, tensor_size(a) * sizeof(f32));
        }
        for (u64 i = 0; i < tensor_size(a); i++) {
            a->grad->data[i] += da->data[i];
        }
        tensor_destroy(b_t);
        tensor_destroy(da);
    }

    if (b->requires_grad) {
        tensor_t *a_t = tensor_transpose(a);
        tensor_t *db = tensor_matmul(a_t, t->grad);
        if (b->grad == NULL) {
            b->grad = tensor_create(NULL, b->shape, b->ndim, false);
            memset(b->grad->data, 0, tensor_size(b) * sizeof(f32));
        }
        for (u64 i = 0; i < tensor_size(b); i++) {
            b->grad->data[i] += db->data[i];
        }
        tensor_destroy(a_t);
        tensor_destroy(db);
    }
}

// Matrix multiplication
tensor_t *tensor_matmul(tensor_t *a, tensor_t *b) {
    // Basic shape check for 2D matrices
    if (a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0]) {
        return NULL;
    }

    i32 new_shape[] = {a->shape[0], b->shape[1]};
    f32 *new_data = (f32 *)calloc((u64)new_shape[0] * (u64)new_shape[1], sizeof(f32));

    for (i32 i = 0; i < a->shape[0]; i++) {
        for (i32 j = 0; j < b->shape[1]; j++) {
            for (i32 k = 0; k < a->shape[1]; k++) {
                new_data[i * new_shape[1] + j] += a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j];
            }
        }
    }

    bool requires_grad = a->requires_grad || b->requires_grad;
    tensor_t *result = tensor_create(new_data, new_shape, 2, requires_grad);
    free(new_data);

    if (requires_grad) {
        result->grad_fn = matmul_backward;
        result->ctx = (void **)malloc(2 * sizeof(void *));
        result->ctx[0] = a;
        result->ctx[1] = b;
        result->ctx_size = 2;
    }

    return result;
}

// Softmax (for a 1D tensor)
tensor_t *tensor_softmax(tensor_t *a) {
    f32 max_val = a->data[0];
    for (u64 i = 1; i < tensor_size(a); i++) {
        if (a->data[i] > max_val) {
            max_val = a->data[i];
        }
    }

    f32 *new_data = (f32 *)malloc(tensor_size(a) * sizeof(f32));
    f32 sum = 0.0f;
    for (u64 i = 0; i < tensor_size(a); i++) {
        new_data[i] = expf(a->data[i] - max_val);
        sum += new_data[i];
    }

    for (u64 i = 0; i < tensor_size(a); i++) {
        new_data[i] /= sum;
    }

    tensor_t *result = tensor_create(new_data, a->shape, a->ndim, a->requires_grad);
    free(new_data);

    // Backward for softmax is complex and will be added later

    return result;
}

void cross_entropy_backward(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    i32 target_idx = *(i32 *)t->ctx[1];

    if (a->requires_grad) {
        // This is a simplified version that combines softmax and cross-entropy.
        // It assumes the input 'a' is the raw output (logits) of the network.
        f32 *softmax_out = malloc(tensor_size(a) * sizeof(f32));
        f32 max_val = a->data[0];
        for (u64 i = 1; i < tensor_size(a); i++) {
            if (a->data[i] > max_val) {
                max_val = a->data[i];
            }
        }
        f32 sum = 0.0f;
        for (u64 i = 0; i < tensor_size(a); i++) {
            softmax_out[i] = expf(a->data[i] - max_val);
            sum += softmax_out[i];
        }
        for (u64 i = 0; i < tensor_size(a); i++) {
            softmax_out[i] /= sum;
        }

        if (a->grad == NULL) {
            a->grad = tensor_create(NULL, a->shape, a->ndim, false);
            memset(a->grad->data, 0, tensor_size(a) * sizeof(f32));
        }
        for (u64 i = 0; i < tensor_size(a); ++i) {
            f32 grad = softmax_out[i];
            if ((i32)i == target_idx) {
                grad -= 1.0f;
            }
            a->grad->data[i] += grad * t->grad->data[0];
        }
        free(softmax_out);
    }
    free(t->ctx[1]); // free the malloced i32 poi32er
}

// Cross-entropy loss
tensor_t *tensor_cross_entropy(tensor_t *a, i32 target_idx) {
    // This is a simplified version that combines softmax and cross-entropy.
    // It assumes the input 'a' is the raw output (logits) of the network.

    // Softmax part
    f32 max_val = a->data[0];
    for (u64 i = 1; i < tensor_size(a); i++) {
        if (a->data[i] > max_val) {
            max_val = a->data[i];
        }
    }
    f32 sum = 0.0f;
    f32 *softmax_out = malloc(tensor_size(a) * sizeof(f32));
    for (u64 i = 0; i < tensor_size(a); i++) {
        softmax_out[i] = expf(a->data[i] - max_val);
        sum += softmax_out[i];
    }
    for (u64 i = 0; i < tensor_size(a); i++) {
        softmax_out[i] /= sum;
    }

    // Cross-entropy part
    f32 loss_val = -logf(softmax_out[target_idx]);
    free(softmax_out);

    i32 shape[] = {1};
    tensor_t *loss = tensor_create(&loss_val, shape, 1, a->requires_grad);

    if (a->requires_grad) {
        loss->grad_fn = cross_entropy_backward;
        loss->ctx = (void **)malloc(2 * sizeof(void *));
        loss->ctx[0] = a;
        i32 *target_idx_ptr = malloc(sizeof(i32));
        *target_idx_ptr = target_idx;
        loss->ctx[1] = target_idx_ptr;
        loss->ctx_size = 2;
    }

    return loss;
}
