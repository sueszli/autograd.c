#include "activations_backward.h"
#include "tensor.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>

// d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
Tensor *tensor_sigmoid_backward(const Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    Tensor *grad = tensor_create(NULL, t->shape, t->ndim, false);

    for (uint64_t i = 0; i < t->size; i++) {
        float32_t x = t->data[i];
        float32_t sigmoid_x = 1.0f / (1.0f + expf(-x));
        grad->data[i] = sigmoid_x * (1.0f - sigmoid_x);
    }

    return grad;
}

// d/dx relu(x) = 1 if x > 0, else 0
Tensor *tensor_relu_backward(const Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    Tensor *grad = tensor_create(NULL, t->shape, t->ndim, false);

    for (uint64_t i = 0; i < t->size; i++) {
        grad->data[i] = (t->data[i] > 0.0f) ? 1.0f : 0.0f;
    }

    return grad;
}

// d/dx tanh(x) = 1 - tanh(x)^2
Tensor *tensor_tanh_backward(const Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    Tensor *grad = tensor_create(NULL, t->shape, t->ndim, false);

    for (uint64_t i = 0; i < t->size; i++) {
        float32_t tanh_x = tanhf(t->data[i]);
        grad->data[i] = 1.0f - tanh_x * tanh_x;
    }

    return grad;
}

// d/dx gelu(x) approximately = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
Tensor *tensor_gelu_backward(const Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    Tensor *grad = tensor_create(NULL, t->shape, t->ndim, false);

    float32_t sqrt_2_over_pi = sqrtf(2.0f / (float32_t)M_PI);
    float32_t coeff = 0.044715f;

    for (uint64_t i = 0; i < t->size; i++) {
        float32_t x = t->data[i];
        float32_t x2 = x * x;
        float32_t x3 = x2 * x;

        float32_t tanh_arg = sqrt_2_over_pi * (x + coeff * x3);
        float32_t tanh_out = tanhf(tanh_arg);
        float32_t sech_sq = 1.0f - tanh_out * tanh_out;

        float32_t d_tanh_arg = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x2);

        grad->data[i] = 0.5f * (1.0f + tanh_out) + 0.5f * x * sech_sq * d_tanh_arg;
    }

    return grad;
}

// full softmax backward typically requires the upstream gradient
// this simplified version returns only the local gradient diagonal
Tensor *tensor_softmax_backward(const Tensor *t, int64_t dim) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    // normalize dimension index
    int64_t ndim = (int64_t)t->ndim;
    int64_t target_dim = (dim < 0) ? (dim + ndim) : dim;
    assert(target_dim >= 0 && target_dim < ndim && "Invalid dimension");

    Tensor *grad = tensor_create(NULL, t->shape, t->ndim, false);

    // for each slice along target_dim, compute softmax and its derivative
    uint64_t outer_size = 1;
    for (int64_t d = 0; d < target_dim; d++) {
        outer_size *= t->shape[d];
    }

    uint64_t dim_size = t->shape[target_dim];

    uint64_t inner_size = 1;
    for (uint64_t d = (uint64_t)target_dim + 1; d < t->ndim; d++) {
        inner_size *= t->shape[d];
    }

    for (uint64_t outer = 0; outer < outer_size; outer++) {
        for (uint64_t inner = 0; inner < inner_size; inner++) {
            // compute softmax for this slice
            uint64_t base_idx = outer * dim_size * inner_size + inner;

            // find max for numerical stability
            float32_t max_val = -INFINITY;
            for (uint64_t d = 0; d < dim_size; d++) {
                uint64_t idx = base_idx + d * inner_size;
                if (t->data[idx] > max_val) {
                    max_val = t->data[idx];
                }
            }

            // compute exp sum
            float32_t exp_sum = 0.0f;
            for (uint64_t d = 0; d < dim_size; d++) {
                uint64_t idx = base_idx + d * inner_size;
                exp_sum += expf(t->data[idx] - max_val);
            }

            // compute softmax values and diagonal Jacobian
            for (uint64_t d = 0; d < dim_size; d++) {
                uint64_t idx = base_idx + d * inner_size;
                float32_t softmax_val = expf(t->data[idx] - max_val) / exp_sum;
                // diagonal of Jacobian: y_i * (1 - y_i)
                grad->data[idx] = softmax_val * (1.0f - softmax_val);
            }
        }
    }
    return grad;
}
