#include "activations_backward.h"
#include "ops/arithmetic.h"
#include "tensor.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

//
// sigmoid
//

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

void sigmoid_backward(Function *fn, const Tensor *grad_output) {
    assert(fn != NULL);
    assert(grad_output != NULL);
    assert(fn->num_inputs == 1);

    Tensor *input = fn->inputs[0];
    const Tensor *output = fn->output;

    if (input != NULL && input->requires_grad) {
        // grad_input = grad_output * output * (1 - output)
        // where output = sigmoid(input)
        Tensor *local_grad = tensor_create(NULL, output->shape, output->ndim, false);
        for (uint64_t i = 0; i < output->size; i++) {
            float32_t out_val = output->data[i];
            local_grad->data[i] = out_val * (1.0f - out_val);
        }

        Tensor *grad_input = tensor_mul(grad_output, local_grad, true); // disable_grad=true
        tensor_free(local_grad);
        accumulate_grad(input, grad_input);
    }
}

//
// relu
//

Tensor *tensor_relu_backward(const Tensor *t) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    Tensor *grad = tensor_create(NULL, t->shape, t->ndim, false);

    for (uint64_t i = 0; i < t->size; i++) {
        grad->data[i] = (t->data[i] > 0.0f) ? 1.0f : 0.0f;
    }

    return grad;
}

void relu_backward(Function *fn, const Tensor *grad_output) {
    assert(fn != NULL);
    assert(grad_output != NULL);
    assert(fn->num_inputs == 1);

    Tensor *input = fn->inputs[0];

    if (input != NULL && input->requires_grad) {
        // grad_input = grad_output * (input > 0 ? 1 : 0)
        Tensor *local_grad = tensor_create(NULL, input->shape, input->ndim, false);
        for (uint64_t i = 0; i < input->size; i++) {
            local_grad->data[i] = (input->data[i] > 0.0f) ? 1.0f : 0.0f;
        }

        Tensor *grad_input = tensor_mul(grad_output, local_grad, true); // disable_grad=true
        tensor_free(local_grad);
        accumulate_grad(input, grad_input);
    }
}

//
// tanh
//

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

void tanh_backward(Function *fn, const Tensor *grad_output) {
    assert(fn != NULL);
    assert(grad_output != NULL);
    assert(fn->num_inputs == 1);

    Tensor *input = fn->inputs[0];
    const Tensor *output = fn->output;

    if (input != NULL && input->requires_grad) {
        // grad_input = grad_output * (1 - output^2)
        // where output = tanh(input)
        Tensor *local_grad = tensor_create(NULL, output->shape, output->ndim, false);
        for (uint64_t i = 0; i < output->size; i++) {
            float32_t out_val = output->data[i];
            local_grad->data[i] = 1.0f - out_val * out_val;
        }

        Tensor *grad_input = tensor_mul(grad_output, local_grad, true); // disable_grad=true
        tensor_free(local_grad);
        accumulate_grad(input, grad_input);
    }
}

//
// gelu
//

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

void gelu_backward(Function *fn, const Tensor *grad_output) {
    assert(fn != NULL);
    assert(grad_output != NULL);
    assert(fn->num_inputs == 1);

    Tensor *input = fn->inputs[0];

    if (input != NULL && input->requires_grad) {
        // Use the existing tensor_gelu_backward function
        Tensor *local_grad = tensor_gelu_backward(input);
        Tensor *grad_input = tensor_mul(grad_output, local_grad, true); // disable_grad=true
        tensor_free(local_grad);
        accumulate_grad(input, grad_input);
    }
}

//
// softmax
//

Tensor *tensor_softmax_backward(const Tensor *t, int64_t dim) {
    assert(t != NULL);
    assert(t->data != NULL || t->size == 0);

    int64_t ndim = (int64_t)t->ndim;
    int64_t target_dim = (dim < 0) ? (dim + ndim) : dim;
    assert(target_dim >= 0 && target_dim < ndim && "Invalid dimension");

    Tensor *grad = tensor_create(NULL, t->shape, t->ndim, false);

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
            uint64_t base_idx = outer * dim_size * inner_size + inner;

            float32_t max_val = -INFINITY;
            for (uint64_t d = 0; d < dim_size; d++) {
                uint64_t idx = base_idx + d * inner_size;
                if (t->data[idx] > max_val) {
                    max_val = t->data[idx];
                }
            }

            float32_t exp_sum = 0.0f;
            for (uint64_t d = 0; d < dim_size; d++) {
                uint64_t idx = base_idx + d * inner_size;
                exp_sum += expf(t->data[idx] - max_val);
            }

            for (uint64_t d = 0; d < dim_size; d++) {
                uint64_t idx = base_idx + d * inner_size;
                float32_t softmax_val = expf(t->data[idx] - max_val) / exp_sum;
                grad->data[idx] = softmax_val * (1.0f - softmax_val);
            }
        }
    }
    return grad;
}

void softmax_backward(Function *fn, const Tensor *grad_output) {
    assert(fn != NULL);
    assert(grad_output != NULL);
    assert(fn->num_inputs == 1);
    assert(fn->ctx != NULL && "softmax_backward requires context");

    Tensor *input = fn->inputs[0];
    const Tensor *output = fn->output;
    int64_t dim = *(int64_t *)fn->ctx;

    if (input != NULL && input->requires_grad) {
        // softmax backward: grad_input = output * (grad_output - sum(grad_output * output))
        // this is the jacobian-vector product for softmax

        int64_t ndim = (int64_t)output->ndim;
        int64_t target_dim = (dim < 0) ? (dim + ndim) : dim;
        assert(target_dim >= 0 && target_dim < ndim && "Invalid dimension");

        uint64_t outer_size = 1;
        for (int64_t d = 0; d < target_dim; d++) {
            outer_size *= output->shape[d];
        }

        uint64_t dim_size = output->shape[target_dim];

        uint64_t inner_size = 1;
        for (uint64_t d = (uint64_t)target_dim + 1; d < output->ndim; d++) {
            inner_size *= output->shape[d];
        }

        Tensor *grad_input = tensor_create(NULL, input->shape, input->ndim, false);

        for (uint64_t outer = 0; outer < outer_size; outer++) {
            for (uint64_t inner = 0; inner < inner_size; inner++) {
                uint64_t base_idx = outer * dim_size * inner_size + inner;

                // sum(grad_output * output) along the softmax dimension
                float32_t sum_grad_output_output = 0.0f;
                for (uint64_t d = 0; d < dim_size; d++) {
                    uint64_t idx = base_idx + d * inner_size;
                    sum_grad_output_output += grad_output->data[idx] * output->data[idx];
                }

                // grad_input = output * (grad_output - sum)
                for (uint64_t d = 0; d < dim_size; d++) {
                    uint64_t idx = base_idx + d * inner_size;
                    grad_input->data[idx] = output->data[idx] * (grad_output->data[idx] - sum_grad_output_output);
                }
            }
        }

        accumulate_grad(input, grad_input);
    }

    free(fn->ctx);
    fn->ctx = NULL;
}
