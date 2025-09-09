#include "tensor.h"
#include "autograd.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to calculate the total number of elements in a tensor
size_t tensor_size(const Tensor *t) {
    if (t == NULL)
        return 0;
    size_t size = 1;
    for (int i = 0; i < t->ndim; i++) {
        size *= (size_t)t->shape[i];
    }
    return size;
}

// Create a new tensor
Tensor *tensor_create(float *data, int *shape, int ndim, bool requires_grad) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    t->shape = (int *)malloc((size_t)ndim * sizeof(int));
    memcpy(t->shape, shape, (size_t)ndim * sizeof(int));
    t->ndim = ndim;

    size_t size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= (size_t)shape[i];
    }

    t->data = (float *)malloc(size * sizeof(float));
    if (data != NULL) {
        memcpy(t->data, data, size * sizeof(float));
    }

    t->requires_grad = requires_grad;
    t->grad = NULL;
    t->grad_fn = NULL;
    t->ctx = NULL;
    t->ctx_size = 0;

    return t;
}

// Destroy a tensor and its data
void tensor_destroy(Tensor *t) {
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

// Print a tensor (simple implementation)
void tensor_print(const Tensor *t) {
    printf("Tensor (shape: [");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if (i < t->ndim - 1)
            printf(", ");
    }
    printf("], requires_grad: %s)\n", t->requires_grad ? "true" : "false");

    size_t size = tensor_size(t);
    for (size_t i = 0; i < size; i++) {
        printf("%f ", t->data[i]);
    }
    printf("\n");
}

void tensor_zero_grad(Tensor *t) {
    if (t->grad) {
        memset(t->grad->data, 0, tensor_size(t->grad) * sizeof(float));
    }
}

// Backward function for addition
void add_backward(Tensor *t) {
    Tensor *a = (Tensor *)t->ctx[0];
    Tensor *b = (Tensor *)t->ctx[1];

    if (a->requires_grad) {
        for (size_t i = 0; i < tensor_size(a); i++) {
            if (a->grad == NULL) {
                a->grad = tensor_create(NULL, a->shape, a->ndim, false);
                memset(a->grad->data, 0, tensor_size(a) * sizeof(float));
            }
            a->grad->data[i] += t->grad->data[i];
        }
    }
    if (b->requires_grad) {
        for (size_t i = 0; i < tensor_size(b); i++) {
            if (b->grad == NULL) {
                b->grad = tensor_create(NULL, b->shape, b->ndim, false);
                memset(b->grad->data, 0, tensor_size(b) * sizeof(float));
            }
            b->grad->data[i] += t->grad->data[i];
        }
    }
}

// Tensor addition
Tensor *tensor_add(Tensor *a, Tensor *b) {
    // Basic shape check
    if (a->ndim != b->ndim)
        return NULL;
    for (int i = 0; i < a->ndim; ++i) {
        if (a->shape[i] != b->shape[i])
            return NULL;
    }

    float *new_data = (float *)malloc(tensor_size(a) * sizeof(float));
    for (size_t i = 0; i < tensor_size(a); i++) {
        new_data[i] = a->data[i] + b->data[i];
    }

    bool requires_grad = a->requires_grad || b->requires_grad;
    Tensor *result = tensor_create(new_data, a->shape, a->ndim, requires_grad);
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
void mul_backward(Tensor *t) {
    Tensor *a = (Tensor *)t->ctx[0];
    Tensor *b = (Tensor *)t->ctx[1];

    if (a->requires_grad) {
        for (size_t i = 0; i < tensor_size(a); i++) {
            if (a->grad == NULL) {
                a->grad = tensor_create(NULL, a->shape, a->ndim, false);
                memset(a->grad->data, 0, tensor_size(a) * sizeof(float));
            }
            a->grad->data[i] += b->data[i] * t->grad->data[i];
        }
    }
    if (b->requires_grad) {
        for (size_t i = 0; i < tensor_size(b); i++) {
            if (b->grad == NULL) {
                b->grad = tensor_create(NULL, b->shape, b->ndim, false);
                memset(b->grad->data, 0, tensor_size(b) * sizeof(float));
            }
            b->grad->data[i] += a->data[i] * t->grad->data[i];
        }
    }
}

// Tensor multiplication
Tensor *tensor_mul(Tensor *a, Tensor *b) {
    // Basic shape check
    if (a->ndim != b->ndim)
        return NULL;
    for (int i = 0; i < a->ndim; ++i) {
        if (a->shape[i] != b->shape[i])
            return NULL;
    }

    float *new_data = (float *)malloc(tensor_size(a) * sizeof(float));
    for (size_t i = 0; i < tensor_size(a); i++) {
        new_data[i] = a->data[i] * b->data[i];
    }

    bool requires_grad = a->requires_grad || b->requires_grad;
    Tensor *result = tensor_create(new_data, a->shape, a->ndim, requires_grad);
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
void relu_backward(Tensor *t) {
    Tensor *a = (Tensor *)t->ctx[0];
    if (a->requires_grad) {
        for (size_t i = 0; i < tensor_size(a); i++) {
            if (a->grad == NULL) {
                a->grad = tensor_create(NULL, a->shape, a->ndim, false);
                memset(a->grad->data, 0, tensor_size(a) * sizeof(float));
            }
            a->grad->data[i] += (a->data[i] > 0) * t->grad->data[i];
        }
    }
}

// ReLU activation
Tensor *tensor_relu(Tensor *a) {
    float *new_data = (float *)malloc(tensor_size(a) * sizeof(float));
    for (size_t i = 0; i < tensor_size(a); i++) {
        new_data[i] = a->data[i] > 0 ? a->data[i] : 0;
    }

    Tensor *result = tensor_create(new_data, a->shape, a->ndim, a->requires_grad);
    free(new_data);

    if (a->requires_grad) {
        result->grad_fn = relu_backward;
        result->ctx = (void **)malloc(sizeof(void *));
        result->ctx[0] = a;
        result->ctx_size = 1;
    }

    return result;
}

Tensor *tensor_transpose(Tensor *a) {
    if (a->ndim != 2)
        return NULL;
    int new_shape[] = {a->shape[1], a->shape[0]};
    float *new_data = malloc(tensor_size(a) * sizeof(float));

    for (int i = 0; i < a->shape[0]; i++) {
        for (int j = 0; j < a->shape[1]; j++) {
            new_data[j * a->shape[0] + i] = a->data[i * a->shape[1] + j];
        }
    }

    Tensor *result = tensor_create(new_data, new_shape, 2, a->requires_grad);
    free(new_data);
    result->requires_grad = false; // Transpose backward is not implemented for simplicity

    return result;
}

// Backward function for matmul
void matmul_backward(Tensor *t) {
    Tensor *a = (Tensor *)t->ctx[0];
    Tensor *b = (Tensor *)t->ctx[1];

    if (a->requires_grad) {
        Tensor *b_t = tensor_transpose(b);
        Tensor *da = tensor_matmul(t->grad, b_t);
        if (a->grad == NULL) {
            a->grad = tensor_create(NULL, a->shape, a->ndim, false);
            memset(a->grad->data, 0, tensor_size(a) * sizeof(float));
        }
        for (size_t i = 0; i < tensor_size(a); i++) {
            a->grad->data[i] += da->data[i];
        }
        tensor_destroy(b_t);
        tensor_destroy(da);
    }

    if (b->requires_grad) {
        Tensor *a_t = tensor_transpose(a);
        Tensor *db = tensor_matmul(a_t, t->grad);
        if (b->grad == NULL) {
            b->grad = tensor_create(NULL, b->shape, b->ndim, false);
            memset(b->grad->data, 0, tensor_size(b) * sizeof(float));
        }
        for (size_t i = 0; i < tensor_size(b); i++) {
            b->grad->data[i] += db->data[i];
        }
        tensor_destroy(a_t);
        tensor_destroy(db);
    }
}

// Matrix multiplication
Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    // Basic shape check for 2D matrices
    if (a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0]) {
        return NULL;
    }

    int new_shape[] = {a->shape[0], b->shape[1]};
    float *new_data = (float *)calloc((size_t)new_shape[0] * (size_t)new_shape[1], sizeof(float));

    for (int i = 0; i < a->shape[0]; i++) {
        for (int j = 0; j < b->shape[1]; j++) {
            for (int k = 0; k < a->shape[1]; k++) {
                new_data[i * new_shape[1] + j] += a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j];
            }
        }
    }

    bool requires_grad = a->requires_grad || b->requires_grad;
    Tensor *result = tensor_create(new_data, new_shape, 2, requires_grad);
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
Tensor *tensor_softmax(Tensor *a) {
    float max_val = a->data[0];
    for (size_t i = 1; i < tensor_size(a); i++) {
        if (a->data[i] > max_val) {
            max_val = a->data[i];
        }
    }

    float *new_data = (float *)malloc(tensor_size(a) * sizeof(float));
    float sum = 0.0f;
    for (size_t i = 0; i < tensor_size(a); i++) {
        new_data[i] = expf(a->data[i] - max_val);
        sum += new_data[i];
    }

    for (size_t i = 0; i < tensor_size(a); i++) {
        new_data[i] /= sum;
    }

    Tensor *result = tensor_create(new_data, a->shape, a->ndim, a->requires_grad);
    free(new_data);

    // Backward for softmax is complex and will be added later

    return result;
}

void cross_entropy_backward(Tensor *t) {
    Tensor *a = (Tensor *)t->ctx[0];
    int target_idx = *(int *)t->ctx[1];

    if (a->requires_grad) {
        // This is a simplified version that combines softmax and cross-entropy.
        // It assumes the input 'a' is the raw output (logits) of the network.
        float *softmax_out = malloc(tensor_size(a) * sizeof(float));
        float max_val = a->data[0];
        for (size_t i = 1; i < tensor_size(a); i++) {
            if (a->data[i] > max_val) {
                max_val = a->data[i];
            }
        }
        float sum = 0.0f;
        for (size_t i = 0; i < tensor_size(a); i++) {
            softmax_out[i] = expf(a->data[i] - max_val);
            sum += softmax_out[i];
        }
        for (size_t i = 0; i < tensor_size(a); i++) {
            softmax_out[i] /= sum;
        }

        if (a->grad == NULL) {
            a->grad = tensor_create(NULL, a->shape, a->ndim, false);
            memset(a->grad->data, 0, tensor_size(a) * sizeof(float));
        }
        for (size_t i = 0; i < tensor_size(a); ++i) {
            float grad = softmax_out[i];
            if ((int)i == target_idx) {
                grad -= 1.0f;
            }
            a->grad->data[i] += grad * t->grad->data[0];
        }
        free(softmax_out);
    }
    free(t->ctx[1]); // free the malloced int pointer
}

// Cross-entropy loss
Tensor *tensor_cross_entropy(Tensor *a, int target_idx) {
    // This is a simplified version that combines softmax and cross-entropy.
    // It assumes the input 'a' is the raw output (logits) of the network.

    // Softmax part
    float max_val = a->data[0];
    for (size_t i = 1; i < tensor_size(a); i++) {
        if (a->data[i] > max_val) {
            max_val = a->data[i];
        }
    }
    float sum = 0.0f;
    float *softmax_out = malloc(tensor_size(a) * sizeof(float));
    for (size_t i = 0; i < tensor_size(a); i++) {
        softmax_out[i] = expf(a->data[i] - max_val);
        sum += softmax_out[i];
    }
    for (size_t i = 0; i < tensor_size(a); i++) {
        softmax_out[i] /= sum;
    }

    // Cross-entropy part
    float loss_val = -logf(softmax_out[target_idx]);
    free(softmax_out);

    int shape[] = {1};
    Tensor *loss = tensor_create(&loss_val, shape, 1, a->requires_grad);

    if (a->requires_grad) {
        loss->grad_fn = cross_entropy_backward;
        loss->ctx = (void **)malloc(2 * sizeof(void *));
        loss->ctx[0] = a;
        int *target_idx_ptr = malloc(sizeof(int));
        *target_idx_ptr = target_idx;
        loss->ctx[1] = target_idx_ptr;
        loss->ctx_size = 2;
    }

    return loss;
}
