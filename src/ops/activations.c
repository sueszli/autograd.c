#include "activations.h"
#include "autograd.h"
#include "ops/activations_backward.h"
#include "ops/arithmetic.h"
#include "ops/reductions.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

Tensor *tensor_sigmoid(const Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    Tensor *out = tensor_create(NULL, t->shape, t->ndim, t->requires_grad);

    for (uint64_t i = 0; i < t->size; i++) {
        float32_t x = t->data[i];
        if (x > 500.0f) {
            x = 500.0f;
        }
        if (x < -500.0f) {
            x = -500.0f;
        }

        if (x >= 0.0f) {
            out->data[i] = 1.0f / (1.0f + expf(-x));
        } else {
            float32_t ex = expf(x);
            out->data[i] = ex / (1.0f + ex);
        }
    }

    if (out->requires_grad) {
        Function *fn = arena_alloc_function();
        fn->apply = sigmoid_backward;
        fn->output = out;
        fn->num_inputs = 1;
        fn->inputs[0] = (Tensor *)t;
        fn->pending_count = 0;
        fn->ctx = NULL;
        if (t->grad_fn != NULL) {
            t->grad_fn->pending_count++;
        }
        out->grad_fn = fn;
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
        Function *fn = arena_alloc_function();
        fn->apply = relu_backward;
        fn->output = out;
        fn->num_inputs = 1;
        fn->inputs[0] = (Tensor *)t;
        fn->pending_count = 0;
        fn->ctx = NULL;
        if (t->grad_fn != NULL) {
            t->grad_fn->pending_count++;
        }
        out->grad_fn = fn;
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
        Function *fn = arena_alloc_function();
        fn->apply = tanh_backward;
        fn->output = out;
        fn->num_inputs = 1;
        fn->inputs[0] = (Tensor *)t;
        fn->pending_count = 0;
        fn->ctx = NULL;
        if (t->grad_fn != NULL) {
            t->grad_fn->pending_count++;
        }
        out->grad_fn = fn;
    }

    return out;
}

Tensor *tensor_gelu(const Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    Tensor *out = tensor_create(NULL, t->shape, t->ndim, t->requires_grad);

    for (uint64_t i = 0; i < t->size; i++) {
        float32_t x = t->data[i];
        out->data[i] = 0.5f * x * (1.0f + erff(x * 1 / (float32_t)sqrt(2)));
    }

    if (out->requires_grad) {
        Function *fn = arena_alloc_function();
        fn->apply = gelu_backward;
        fn->output = out;
        fn->num_inputs = 1;
        fn->inputs[0] = (Tensor *)t;
        fn->pending_count = 0;
        fn->ctx = NULL;
        if (t->grad_fn != NULL) {
            t->grad_fn->pending_count++;
        }
        out->grad_fn = fn;
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
        Function *fn = arena_alloc_function();
        fn->apply = softmax_backward;
        fn->output = out;
        fn->num_inputs = 1;
        fn->inputs[0] = (Tensor *)t;
        fn->pending_count = 0;

        // store dimension in context
        int64_t *ctx = (int64_t *)malloc(sizeof(int64_t));
        assert(ctx != NULL && "malloc failed");
        *ctx = dim;
        fn->ctx = ctx;

        if (t->grad_fn != NULL) {
            t->grad_fn->pending_count++;
        }
        out->grad_fn = fn;
    }

    return out;
}
