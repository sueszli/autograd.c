#include "ops.h"
#include "../utils/defer.h"
#include "../utils/types.h"
#include "broadcast.h"
#include "tensor.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    TENSOR_OP_ADD,
    TENSOR_OP_SUB,
    TENSOR_OP_MUL,
    TENSOR_OP_DIV,
} tensor_op_type_t;

static bool shapes_match(const tensor_t *a, const tensor_t *b) {
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

static void reduce_gradient_if_needed(tensor_t *grad, const tensor_t *original_tensor) {
    if (shapes_match(grad, original_tensor)) {
        return;
    }

    u64 grad_size = tensor_size(grad);
    u64 original_size = tensor_size(original_tensor);
    f32 *reduced_data = (f32 *)calloc(original_size, sizeof(f32));
    assert(reduced_data != NULL);

    for (u64 i = 0; i < grad_size; i++) {
        reduced_data[i % original_size] += grad->data[i];
    }

    free(grad->data);
    grad->data = reduced_data;

    free(grad->shape);
    grad->shape = (i32 *)malloc((size_t)original_tensor->ndim * sizeof(i32));
    assert(grad->shape != NULL);
    memcpy(grad->shape, original_tensor->shape, (size_t)original_tensor->ndim * sizeof(i32));
    grad->ndim = original_tensor->ndim;
}

static void accumulate_gradient(tensor_t *tensor, tensor_t *grad_update, bool needs_broadcasting) {
    if (!tensor->requires_grad) {
        return;
    }

    if (tensor->grad == NULL) {
        tensor->grad = tensor_create(NULL, grad_update->shape, grad_update->ndim, false);
        tensor_zero_grad(tensor);
    }

    for (u64 i = 0; i < tensor_size(grad_update); i++) {
        tensor->grad->data[i] += grad_update->data[i];
    }

    if (needs_broadcasting) {
        reduce_gradient_if_needed(tensor->grad, tensor);
    }
}

static tensor_t *tensor_op_execution(tensor_t *a, tensor_t *b, tensor_op_type_t op_type) {
    assert(a != NULL && b != NULL);

    bool needs_broadcasting = !shapes_match(a, b);
    tensor_t *op_a = a;
    tensor_t *op_b = b;

    if (needs_broadcasting) {
        broadcasted_tensors_t broadcasted = tensor_broadcast(a, b);
        op_a = broadcasted.a;
        op_b = broadcasted.b;
    } else {
        assert(tensor_size(a) == tensor_size(b));
    }

    defer({
        if (op_a != a)
            tensor_destroy(op_a);
        if (op_b != b)
            tensor_destroy(op_b);
    });

    u64 size = tensor_size(op_a);
    f32 *new_data = (f32 *)malloc(size * sizeof(f32));
    assert(new_data != NULL);

    for (u64 i = 0; i < size; i++) {
        switch (op_type) {
        case TENSOR_OP_ADD:
            new_data[i] = op_a->data[i] + op_b->data[i];
            break;
        case TENSOR_OP_SUB:
            new_data[i] = op_a->data[i] - op_b->data[i];
            break;
        case TENSOR_OP_MUL:
            new_data[i] = op_a->data[i] * op_b->data[i];
            break;
        case TENSOR_OP_DIV:
            new_data[i] = op_a->data[i] / op_b->data[i];
            break;
        }
    }

    tensor_t *result = tensor_create(new_data, op_a->shape, op_a->ndim, false);
    free(new_data);

    return result;
}

static void add_backward(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool needs_broadcasting = (bool)(intptr_t)t->ctx[2];

    accumulate_gradient(a, t->grad, needs_broadcasting);
    accumulate_gradient(b, t->grad, needs_broadcasting);
}

static void sub_backward(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool needs_broadcasting = (bool)(intptr_t)t->ctx[2];

    accumulate_gradient(a, t->grad, needs_broadcasting);

    tensor_t *neg_grad = tensor_create(NULL, t->grad->shape, t->grad->ndim, false);
    for (u64 i = 0; i < tensor_size(t->grad); i++) {
        neg_grad->data[i] = -t->grad->data[i];
    }
    accumulate_gradient(b, neg_grad, needs_broadcasting);
    tensor_destroy(neg_grad);
}

static void mul_backward(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool needs_broadcasting = (bool)(intptr_t)t->ctx[2];

    tensor_t *grad_a = tensor_op_execution(t->grad, b, TENSOR_OP_MUL);
    tensor_t *grad_b = tensor_op_execution(t->grad, a, TENSOR_OP_MUL);
    defer({
        tensor_destroy(grad_a);
        tensor_destroy(grad_b);
    });

    accumulate_gradient(a, grad_a, needs_broadcasting);
    accumulate_gradient(b, grad_b, needs_broadcasting);
}

static void div_backward(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool needs_broadcasting = (bool)(intptr_t)t->ctx[2];

    tensor_t *grad_a = tensor_op_execution(t->grad, b, TENSOR_OP_DIV);

    tensor_t *b_squared = tensor_op_execution(b, b, TENSOR_OP_MUL);
    tensor_t *a_mul_grad = tensor_op_execution(a, t->grad, TENSOR_OP_MUL);
    tensor_t *numerator = tensor_op_execution(a_mul_grad, t->grad, TENSOR_OP_MUL);
    tensor_t *grad_b = tensor_op_execution(numerator, b_squared, TENSOR_OP_DIV);
    tensor_t *neg_grad_b = tensor_create(NULL, grad_b->shape, grad_b->ndim, false);
    for (u64 i = 0; i < tensor_size(grad_b); i++) {
        neg_grad_b->data[i] = -grad_b->data[i];
    }

    defer({
        tensor_destroy(grad_a);
        tensor_destroy(b_squared);
        tensor_destroy(a_mul_grad);
        tensor_destroy(numerator);
        tensor_destroy(grad_b);
        tensor_destroy(neg_grad_b);
    });

    accumulate_gradient(a, grad_a, needs_broadcasting);
    accumulate_gradient(b, neg_grad_b, needs_broadcasting);
}

static tensor_t *tensor_op(tensor_t *a, tensor_t *b, tensor_op_type_t op_type) {
    tensor_t *result = tensor_op_execution(a, b, op_type);

    bool requires_grad = a->requires_grad || b->requires_grad;
    if (requires_grad) {
        result->requires_grad = true;
        result->ctx = (void **)malloc(3 * sizeof(void *));
        assert(result->ctx != NULL);
        result->ctx[0] = a;
        result->ctx[1] = b;
        bool needs_broadcasting = !shapes_match(a, b);
        result->ctx[2] = (void *)(intptr_t)needs_broadcasting;
        result->ctx_size = 3;

        switch (op_type) {
        case TENSOR_OP_ADD:
            result->grad_fn = add_backward;
            break;
        case TENSOR_OP_SUB:
            result->grad_fn = sub_backward;
            break;
        case TENSOR_OP_MUL:
            result->grad_fn = mul_backward;
            break;
        case TENSOR_OP_DIV:
            result->grad_fn = div_backward;
            break;
        }
    }

    return result;
}

tensor_t *tensor_add(tensor_t *a, tensor_t *b) { return tensor_op(a, b, TENSOR_OP_ADD); }
tensor_t *tensor_sub(tensor_t *a, tensor_t *b) { return tensor_op(a, b, TENSOR_OP_SUB); }
tensor_t *tensor_mul(tensor_t *a, tensor_t *b) { return tensor_op(a, b, TENSOR_OP_MUL); }
tensor_t *tensor_div(tensor_t *a, tensor_t *b) { return tensor_op(a, b, TENSOR_OP_DIV); }

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

void conv2d_backward(tensor_t *t) {
    // TODO: Implement backward pass for conv2d
}

tensor_t *tensor_conv2d(tensor_t *input, tensor_t *kernel, i32 stride, i32 padding) {
    assert(input->ndim == 2);
    assert(kernel->ndim == 2);

    i32 input_height = input->shape[0];
    i32 input_width = input->shape[1];
    i32 kernel_height = kernel->shape[0];
    i32 kernel_width = kernel->shape[1];

    i32 out_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    i32 out_width = (input_width - kernel_width + 2 * padding) / stride + 1;

    i32 out_shape[] = {out_height, out_width};
    f32 *out_data = (f32 *)calloc((u64)out_height * (u64)out_width, sizeof(f32));
    assert(out_data != NULL);

    for (i32 i = 0; i < out_height; i++) {
        for (i32 j = 0; j < out_width; j++) {
            f32 sum = 0.0f;
            for (i32 ki = 0; ki < kernel_height; ki++) {
                for (i32 kj = 0; kj < kernel_width; kj++) {
                    i32 row = i * stride + ki - padding;
                    i32 col = j * stride + kj - padding;
                    if (row >= 0 && row < input_height && col >= 0 && col < input_width) {
                        sum += input->data[row * input_width + col] * kernel->data[ki * kernel_width + kj];
                    }
                }
            }
            out_data[i * out_width + j] = sum;
        }
    }

    bool requires_grad = input->requires_grad || kernel->requires_grad;
    tensor_t *result = tensor_create(out_data, out_shape, 2, requires_grad);
    free(out_data);

    if (requires_grad) {
        result->grad_fn = conv2d_backward;
        result->ctx = (void **)malloc(2 * sizeof(void *));
        result->ctx[0] = input;
        result->ctx[1] = kernel;
        result->ctx_size = 2;
    }

    return result;
}

void max_pool2d_backward(tensor_t *t) {
    // TODO: Implement backward pass for max_pool2d
}

tensor_t *tensor_max_pool2d(tensor_t *input, i32 kernel_size, i32 stride) {
    assert(input->ndim == 2);

    i32 input_height = input->shape[0];
    i32 input_width = input->shape[1];

    i32 out_height = (input_height - kernel_size) / stride + 1;
    i32 out_width = (input_width - kernel_size) / stride + 1;

    i32 out_shape[] = {out_height, out_width};
    f32 *out_data = (f32 *)calloc((u64)out_height * (u64)out_width, sizeof(f32));
    assert(out_data != NULL);

    for (i32 i = 0; i < out_height; i++) {
        for (i32 j = 0; j < out_width; j++) {
            f32 max_val = -__FLT_MAX__;
            for (i32 ki = 0; ki < kernel_size; ki++) {
                for (i32 kj = 0; kj < kernel_size; kj++) {
                    i32 row = i * stride + ki;
                    i32 col = j * stride + kj;
                    if (input->data[row * input_width + col] > max_val) {
                        max_val = input->data[row * input_width + col];
                    }
                }
            }
            out_data[i * out_width + j] = max_val;
        }
    }

    bool requires_grad = input->requires_grad;
    tensor_t *result = tensor_create(out_data, out_shape, 2, requires_grad);
    free(out_data);

    if (requires_grad) {
        result->grad_fn = max_pool2d_backward;
        result->ctx = (void **)malloc(1 * sizeof(void *));
        result->ctx[0] = input;
        result->ctx_size = 1;
    }

    return result;
}

void avg_pool2d_backward(tensor_t *t) {
    // TODO: Implement backward pass for avg_pool2d
}

tensor_t *tensor_avg_pool2d(tensor_t *input, i32 kernel_size, i32 stride) {
    assert(input->ndim == 2);

    i32 input_height = input->shape[0];
    i32 input_width = input->shape[1];

    i32 out_height = (input_height - kernel_size) / stride + 1;
    i32 out_width = (input_width - kernel_size) / stride + 1;

    i32 out_shape[] = {out_height, out_width};
    f32 *out_data = (f32 *)calloc((u64)out_height * (u64)out_width, sizeof(f32));
    assert(out_data != NULL);

    for (i32 i = 0; i < out_height; i++) {
        for (i32 j = 0; j < out_width; j++) {
            f32 sum = 0.0f;
            for (i32 ki = 0; ki < kernel_size; ki++) {
                for (i32 kj = 0; kj < kernel_size; kj++) {
                    i32 row = i * stride + ki;
                    i32 col = j * stride + kj;
                    sum += input->data[row * input_width + col];
                }
            }
            out_data[i * out_width + j] = sum / (f32)(kernel_size * kernel_size);
        }
    }

    bool requires_grad = input->requires_grad;
    tensor_t *result = tensor_create(out_data, out_shape, 2, requires_grad);
    free(out_data);

    if (requires_grad) {
        result->grad_fn = avg_pool2d_backward;
        result->ctx = (void **)malloc(1 * sizeof(void *));
        result->ctx[0] = input;
        result->ctx_size = 1;
    }

    return result;
}
