#include "activations.h"
#include "autograd.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>

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
    return out;
}

Tensor *tensor_tanh(const Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    Tensor *out = tensor_create(NULL, t->shape, t->ndim, t->requires_grad);

    for (uint64_t i = 0; i < t->size; i++) {
        out->data[i] = tanhf(t->data[i]);
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
    return out;
}
