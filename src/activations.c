#include "activations.h"
#include "autograd.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>

static inline float32_t sigmoid(float32_t x) {
    if (x > 500.0f) {
        x = 500.0f;
    }
    if (x < -500.0f) {
        x = -500.0f;
    }

    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        float32_t ex = expf(x);
        return ex / (1.0f + ex);
    }
}

Tensor *tensor_sigmoid(const Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    Tensor *out = tensor_create(NULL, t->shape, t->ndim, t->requires_grad);

    for (uint64_t i = 0; i < t->size; i++) {
        out->data[i] = sigmoid(t->data[i]);
    }
    if (out->requires_grad) {
        out->grad_fn = new_sigmoid_backward((Tensor *)t, out);
        out->grad_fn->out_tensor = out;
    }
    return out;
}

Tensor *tensor_relu(const Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    Tensor *out = tensor_create(NULL, t->shape, t->ndim, t->requires_grad);

    for (uint64_t i = 0; i < t->size; i++) {
        float32_t x = t->data[i];
        out->data[i] = (x > 0.0f) ? x : 0.0f;
    }
    if (out->requires_grad) {
        out->grad_fn = new_relu_backward((Tensor *)t);
        out->grad_fn->out_tensor = out;
    }
    return out;
}

Tensor *tensor_tanh(const Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    Tensor *out = tensor_create(NULL, t->shape, t->ndim, t->requires_grad);

    for (uint64_t i = 0; i < t->size; i++) {
        out->data[i] = tanhf(t->data[i]);
    }
    if (out->requires_grad) {
        out->grad_fn = new_tanh_backward((Tensor *)t, out);
        out->grad_fn->out_tensor = out;
    }
    return out;
}

Tensor *tensor_gelu(const Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    Tensor *out = tensor_create(NULL, t->shape, t->ndim, t->requires_grad);

    for (uint64_t i = 0; i < t->size; i++) {
        float32_t x = t->data[i];
        out->data[i] = 0.5f * x * (1.0f + erff(x * 1 / (float)sqrt(2)));
    }

    if (out->requires_grad) {
        out->grad_fn = new_gelu_backward((Tensor *)t);
        out->grad_fn->out_tensor = out;
    }
    return out;
}

Tensor *tensor_softmax(const Tensor *t, int64_t dim) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    Tensor *max_val = tensor_max(t, dim, true);
    Tensor *shifted = tensor_sub(t, max_val);
    tensor_free(max_val);

    for (uint64_t i = 0; i < shifted->size; i++) {
        shifted->data[i] = expf(shifted->data[i]);
    }

    Tensor *sum_exp = tensor_sum(shifted, dim, true);
    Tensor *out = tensor_div(shifted, sum_exp);

    tensor_free(shifted);
    tensor_free(sum_exp);

    if (out->requires_grad) {
        // Free the DivBackward grad_fn created by tensor_div
        // This is necessary because we are replacing it with SoftmaxBackward,
        // and DivBackward holds dangling pointers to shifted and sum_exp (which are freed).
        if (out->grad_fn) {
            grad_fn_free(out->grad_fn);
        }
        out->grad_fn = new_softmax_backward((Tensor *)t, out, dim);
        out->grad_fn->out_tensor = out;
    }
    return out;
}
