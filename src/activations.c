#include "activations.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>

// Sigmoid: 1 / (1 + exp(-x))
// cppcheck-suppress unusedFunction
Tensor *tensor_sigmoid(const Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    Tensor *out = tensor_create(NULL, t->shape, t->ndim, t->requires_grad);

    for (uint64_t i = 0; i < t->size; i++) {
        float32_t x = t->data[i];

        // clip to avoid overflow/underflow in exp
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

// ReLU: max(0, x)
// cppcheck-suppress unusedFunction
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

// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
// Uses standard tanhf
// cppcheck-suppress unusedFunction
Tensor *tensor_tanh(const Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    Tensor *out = tensor_create(NULL, t->shape, t->ndim, t->requires_grad);

    for (uint64_t i = 0; i < t->size; i++) {
        out->data[i] = tanhf(t->data[i]);
    }
    return out;
}

// GELU: x * sigmoid(1.702 * x)
// cppcheck-suppress unusedFunction
Tensor *tensor_gelu(const Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    Tensor *out = tensor_create(NULL, t->shape, t->ndim, t->requires_grad);

    for (uint64_t i = 0; i < t->size; i++) {
        float32_t x = t->data[i];
        float32_t val = 1.702f * x;

        // clip to avoid overflow/underflow in exp
        float32_t sigmoid_val;
        if (val > 500.0f) {
            val = 500.0f;
        }
        if (val < -500.0f) {
            val = -500.0f;
        }

        if (val >= 0.0f) {
            sigmoid_val = 1.0f / (1.0f + expf(-val));
        } else {
            float32_t ex = expf(val);
            sigmoid_val = ex / (1.0f + ex);
        }

        out->data[i] = x * sigmoid_val;
    }
    return out;
}

// Softmax: exp(x_i) / sum(exp(x_j))
// cppcheck-suppress unusedFunction
Tensor *tensor_softmax(const Tensor *t, int64_t dim) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    // subtract max for numerical stability
    // `keepdims=true` so we can broadcast subtract
    Tensor *max_val = tensor_max(t, dim, true);
    Tensor *shifted = tensor_sub(t, max_val);
    tensor_free(max_val);

    // exponentiate shifted values (in place since shifted is a new tensor)
    for (uint64_t i = 0; i < shifted->size; i++) {
        shifted->data[i] = expf(shifted->data[i]);
    }

    // sum exponentials
    Tensor *sum_exp = tensor_sum(shifted, dim, true);

    // divide exponentials by sum
    Tensor *out = tensor_div(shifted, sum_exp);

    tensor_free(shifted);
    tensor_free(sum_exp);

    return out;
}
