#include "activations.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

/*
 * Sigmoid maps any real number to the range (0, 1).
 *
 * Sigmoid Curve:
 *     1.0 ┤     ╭─────
 *         │    ╱
 *     0.5 ┤   ╱
 *         │  ╱
 *     0.0 ┤─╱─────────
 *        -3  0  3
 */
Tensor *tensor_sigmoid(Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL);

    Tensor *out = tensor_zeros(t->shape, t->ndim, t->requires_grad);
    uint64_t size = t->size;

    for (uint64_t i = 0; i < size; i++) {
        float32_t x = t->data[i];

        // Clip to avoid overflow/underflow
        // sigmoid(-500) approx 0, sigmoid(500) approx 1
        if (x > 500.0f) x = 500.0f;
        if (x < -500.0f) x = -500.0f;

        // Numerically stable sigmoid
        // For positive values: 1 / (1 + exp(-x))
        // For negative values: exp(x) / (1 + exp(x))
        if (x >= 0.0f) {
            out->data[i] = 1.0f / (1.0f + expf(-x));
        } else {
            float32_t exp_x = expf(x);
            out->data[i] = exp_x / (1.0f + exp_x);
        }
    }

    return out;
}

/*
 * ReLU sets negative values to zero, keeps positive values unchanged.
 *
 * ReLU Function:
 *         ╱
 *     2  ╱
 *       ╱
 *     1╱
 *     ╱
 *    ╱
 *   ╱
 * ─┴─────
 * -2  0  2
 */
Tensor *tensor_relu(Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL);

    Tensor *out = tensor_zeros(t->shape, t->ndim, t->requires_grad);
    uint64_t size = t->size;

    for (uint64_t i = 0; i < size; i++) {
        float32_t x = t->data[i];
        out->data[i] = (x > 0.0f) ? x : 0.0f;
    }

    return out;
}

/*
 * Tanh maps any real number to (-1, 1) range.
 *
 * Tanh Curve:
 *     1 ┤     ╭─────
 *       │    ╱
 *     0 ┤───╱─────
 *       │  ╱
 *    -1 ┤─╱───────
 *      -3  0  3
 */
Tensor *tensor_tanh(Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL);

    Tensor *out = tensor_zeros(t->shape, t->ndim, t->requires_grad);
    uint64_t size = t->size;

    for (uint64_t i = 0; i < size; i++) {
        out->data[i] = tanhf(t->data[i]);
    }

    return out;
}

/*
 * GELU approximation: x * sigmoid(1.702 * x)
 *
 * GELU Function:
 *         ╱
 *     1  ╱
 *       ╱
 *      ╱
 *     ╱
 *    ╱
 *   ╱ ↙ (smooth curve, no sharp corner)
 *  ╱
 * ─┴─────
 * -2  0  2
 */
Tensor *tensor_gelu(Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL);

    Tensor *out = tensor_zeros(t->shape, t->ndim, t->requires_grad);
    uint64_t size = t->size;
    float32_t coeff = 1.702f;

    for (uint64_t i = 0; i < size; i++) {
        float32_t x = t->data[i];
        float32_t val = coeff * x;

        // Sigmoid of (1.702 * x)
        // Clip to avoid overflow/underflow
        if (val > 500.0f) val = 500.0f;
        if (val < -500.0f) val = -500.0f;

        float32_t sigmoid_part;
        if (val >= 0.0f) {
            sigmoid_part = 1.0f / (1.0f + expf(-val));
        } else {
            float32_t exp_val = expf(val);
            sigmoid_part = exp_val / (1.0f + exp_val);
        }

        out->data[i] = x * sigmoid_part;
    }

    return out;
}

/*
 * Softmax converts any vector to a probability distribution.
 *
 * f(x_i) = e^(x_i) / Σ(e^(x_j))
 */
Tensor *tensor_softmax(Tensor *t, int64_t dim) {
    assert(t != NULL);
    assert(t->data != NULL);

    // Handle negative dimension index
    if (dim < 0) {
        dim += (int64_t)t->ndim;
    }
    assert(dim >= 0 && dim < (int64_t)t->ndim);

    // 1. x_max = max(x, dim, keepdims=True)
    // 2. x_shifted = x - x_max
    // 3. exp_values = exp(x_shifted)
    // 4. exp_sum = sum(exp_values, dim, keepdims=True)
    // 5. result = exp_values / exp_sum

    Tensor *max_val = tensor_max(t, dim, true);
    Tensor *shifted = tensor_sub(t, max_val);

    // We need to compute exp element-wise. tensor.h doesn't seem to have tensor_exp.
    // We can do it manually on 'shifted'.
    Tensor *exp_values = tensor_zeros(shifted->shape, shifted->ndim, shifted->requires_grad);
    for (uint64_t i = 0; i < shifted->size; i++) {
        exp_values->data[i] = expf(shifted->data[i]);
    }

    Tensor *sum_exp = tensor_sum(exp_values, dim, true);
    Tensor *out = tensor_div(exp_values, sum_exp);

    // Clean up intermediate tensors
    tensor_free(max_val);
    tensor_free(shifted);
    tensor_free(exp_values);
    tensor_free(sum_exp);

    return out;
}
