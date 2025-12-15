#include "tensor_backward.h"
#include "tensor.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NDIM 32
#define MAX_TENSOR_SIZE 100000000

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef EPSILON
#define EPSILON 1e-7f
#endif

//
// Helpers
//

static Tensor *unbroadcast(const Tensor *grad, const Tensor *input) {
    if (!grad || !input) {
        return NULL;
    }

    const Tensor *current_grad = grad;
    bool owns_current_grad = false;

    // broadcasting adds dimensions on the left, so collapse extra leading dimensions
    while (current_grad->ndim > input->ndim) {
        Tensor *summed = tensor_sum(current_grad, 0, false);
        if (owns_current_grad) {
            tensor_free((Tensor *)current_grad);
        }
        current_grad = summed;
        owns_current_grad = true;
    }

    assert(current_grad->ndim == input->ndim);
    assert(input->ndim < MAX_NDIM && "Number of dimensions exceeds maximum");

    for (uint64_t dim_idx = 0; dim_idx < input->ndim; dim_idx++) {
        // dimension was broadcasted from 1 to N, so sum back to 1
        if (input->shape[dim_idx] == 1 && current_grad->shape[dim_idx] > 1) {
            Tensor *summed = tensor_sum(current_grad, (int64_t)dim_idx, true);
            if (owns_current_grad) {
                tensor_free((Tensor *)current_grad);
            }
            current_grad = summed;
            owns_current_grad = true;
        }
    }

    if (!owns_current_grad) {
        return tensor_create(grad->data, grad->shape, grad->ndim, false);
    }

    return (Tensor *)current_grad;
}

//
// Arithmetic
//

void tensor_add_backward(const Tensor *grad_output, const Tensor *a, const Tensor *b, Tensor **out_grad_a, Tensor **out_grad_b) {
    assert(grad_output != NULL);
    assert(a != NULL);
    assert(b != NULL);

    if (out_grad_a) *out_grad_a = unbroadcast(grad_output, a);
    if (out_grad_b) *out_grad_b = unbroadcast(grad_output, b);
}

void tensor_sub_backward(const Tensor *grad_output, const Tensor *a, const Tensor *b, Tensor **out_grad_a, Tensor **out_grad_b) {
    assert(grad_output != NULL);
    assert(a != NULL);
    assert(b != NULL);

    if (out_grad_a) *out_grad_a = unbroadcast(grad_output, a);
    if (out_grad_b) {
        Tensor *zeros = tensor_zeros(grad_output->shape, grad_output->ndim, false);
        Tensor *neg_grad = tensor_sub(zeros, grad_output);
        tensor_free(zeros);
        *out_grad_b = unbroadcast(neg_grad, b);
        tensor_free(neg_grad);
    }
}

void tensor_mul_backward(const Tensor *grad_output, const Tensor *a, const Tensor *b, Tensor **out_grad_a, Tensor **out_grad_b) {
    assert(grad_output != NULL);
    assert(a != NULL);
    assert(b != NULL);

    if (out_grad_a) {
        Tensor *da = tensor_mul(grad_output, b);
        *out_grad_a = unbroadcast(da, a);
        tensor_free(da);
    }
    if (out_grad_b) {
        Tensor *db = tensor_mul(grad_output, a);
        *out_grad_b = unbroadcast(db, b);
        tensor_free(db);
    }
}

void tensor_div_backward(const Tensor *grad_output, const Tensor *a, const Tensor *b, Tensor **out_grad_a, Tensor **out_grad_b) {
    assert(grad_output != NULL);
    assert(a != NULL);
    assert(b != NULL);

    if (out_grad_a) {
        Tensor *da = tensor_div(grad_output, b);
        *out_grad_a = unbroadcast(da, a);
        tensor_free(da);
    }
    if (out_grad_b) {
        // -dOut * A / B^2
        Tensor *zeros = tensor_zeros(grad_output->shape, grad_output->ndim, false);
        Tensor *neg_grad = tensor_sub(zeros, grad_output);
        tensor_free(zeros);

        Tensor *num = tensor_mul(neg_grad, a);
        tensor_free(neg_grad);

        Tensor *b_sq = tensor_mul(b, b);
        Tensor *db = tensor_div(num, b_sq);
        tensor_free(num);
        tensor_free(b_sq);

        *out_grad_b = unbroadcast(db, b);
        tensor_free(db);
    }
}

void tensor_matmul_backward(const Tensor *grad_output, const Tensor *a, const Tensor *b, Tensor **out_grad_a, Tensor **out_grad_b) {
    assert(grad_output != NULL);
    assert(a != NULL);
    assert(b != NULL);

    if (out_grad_a) {
        Tensor *b_T = tensor_transpose(b, 0, 1);
        *out_grad_a = tensor_matmul(grad_output, b_T);
        tensor_free(b_T);
    }
    if (out_grad_b) {
        Tensor *a_T = tensor_transpose(a, 0, 1);
        *out_grad_b = tensor_matmul(a_T, grad_output);
        tensor_free(a_T);
    }
}

//
// Reductions
//

Tensor *tensor_sum_backward(const Tensor *grad_output, const Tensor *input, int64_t dim_idx, bool keepdims) {
    assert(grad_output != NULL);
    assert(input != NULL);

    const Tensor *grad_expanded = grad_output;
    bool needs_free = false;

    if (!keepdims) {
        int64_t ndim = (int64_t)input->ndim;
        int64_t new_shape[MAX_NDIM] = {0};

        int64_t target_dim = (dim_idx < 0) ? (dim_idx + ndim) : dim_idx;
        assert(target_dim >= 0 && target_dim < ndim);

        int64_t grad_dim_idx = 0;
        for (int64_t dim = 0; dim < ndim; dim++) {
            if (dim == target_dim) {
                new_shape[dim] = 1;
            } else {
                new_shape[dim] = (int64_t)grad_output->shape[grad_dim_idx++];
            }
        }

        grad_expanded = tensor_reshape(grad_output, new_shape, (uint64_t)ndim);
        needs_free = true;
    }

    // clone to ensure ownership
    Tensor *result = tensor_create(grad_expanded->data, grad_expanded->shape, grad_expanded->ndim, false);

    if (needs_free) {
        tensor_free((Tensor *)grad_expanded);
    }
    return result;
}

Tensor *tensor_mean_backward(const Tensor *grad_output, const Tensor *input, int64_t dim_idx, bool keepdims) {
    assert(grad_output != NULL);
    assert(input != NULL);

    // Re-use sum backward logic for shape expansion
    Tensor *grad_sum = tensor_sum_backward(grad_output, input, dim_idx, keepdims);

    // Scale by 1/N
    int64_t ndim = (int64_t)input->ndim;
    int64_t target_dim = (dim_idx < 0) ? (dim_idx + ndim) : dim_idx;
    uint64_t dim_size = input->shape[target_dim];

    Tensor *dim_size_tensor = tensor_zeros(grad_sum->shape, grad_sum->ndim, false);
    for (uint64_t i = 0; i < dim_size_tensor->size; i++)
        dim_size_tensor->data[i] = (float32_t)dim_size;

    Tensor *result = tensor_div(grad_sum, dim_size_tensor);

    tensor_free(dim_size_tensor);
    tensor_free(grad_sum);

    return result;
}

Tensor *tensor_max_backward(const Tensor *grad_output, const Tensor *input, const Tensor *output, int64_t dim_idx, bool keepdims) {
    assert(grad_output != NULL);
    assert(input != NULL);
    assert(output != NULL);

    // Expand grad and output to match input shape
    const Tensor *grad_expanded = grad_output;
    bool free_grad = false;
    const Tensor *output_expanded = output;
    bool free_output = false;

    if (!keepdims) {
        int64_t ndim = (int64_t)input->ndim;
        int64_t new_shape[MAX_NDIM] = {0};
        int64_t target_dim = (dim_idx < 0) ? (dim_idx + ndim) : dim_idx;

        int64_t g_idx = 0;
        int64_t o_idx = 0;
        for (int64_t dim = 0; dim < ndim; dim++) {
            if (dim == target_dim) {
                new_shape[dim] = 1;
            } else {
                new_shape[dim] = (int64_t)grad_output->shape[g_idx++];
            }
        }
        grad_expanded = tensor_reshape(grad_output, new_shape, (uint64_t)ndim);
        free_grad = true;

        // reuse new_shape logic for output
        g_idx = 0; // reset index logic but use output shape
        for (int64_t dim = 0; dim < ndim; dim++) {
             if (dim == target_dim) {
                new_shape[dim] = 1;
            } else {
                new_shape[dim] = (int64_t)output->shape[o_idx++];
            }
        }
        output_expanded = tensor_reshape(output, new_shape, (uint64_t)ndim);
        free_output = true;
    }

    Tensor *grad_input = tensor_zeros(input->shape, input->ndim, false);

    for (uint64_t elem_idx = 0; elem_idx < input->size; elem_idx++) {
        // Calculate broadcast index
        uint64_t remaining = elem_idx;
        uint64_t broadcast_idx = 0;
        uint64_t broadcast_stride = 1;

        for (int64_t dim = (int64_t)input->ndim - 1; dim >= 0; dim--) {
            uint64_t idx_in_dim = remaining % input->shape[dim];
            remaining /= input->shape[dim];

            uint64_t out_idx_in_dim = (output_expanded->shape[dim] == 1) ? 0 : idx_in_dim;
            broadcast_idx += out_idx_in_dim * broadcast_stride;
            broadcast_stride *= output_expanded->shape[dim];
        }

        if (fabsf(input->data[elem_idx] - output_expanded->data[broadcast_idx]) < EPSILON) {
            grad_input->data[elem_idx] = grad_expanded->data[broadcast_idx];
        }
    }

    if (free_grad) tensor_free((Tensor *)grad_expanded);
    if (free_output) tensor_free((Tensor *)output_expanded);

    return grad_input;
}

//
// Activations
//

Tensor *tensor_relu_backward(const Tensor *grad_output, const Tensor *input) {
    assert(grad_output != NULL);
    assert(input != NULL);
    assert(grad_output->size == input->size);

    Tensor *grad_input = tensor_zeros(grad_output->shape, grad_output->ndim, false);
    for (uint64_t i = 0; i < grad_output->size; i++) {
        grad_input->data[i] = (input->data[i] > 0.0f) ? grad_output->data[i] : 0.0f;
    }
    return grad_input;
}

Tensor *tensor_sigmoid_backward(const Tensor *grad_output, const Tensor *output) {
    assert(grad_output != NULL);
    assert(output != NULL);

    // grad_input = grad_output * output * (1 - output)
    Tensor *ones = tensor_zeros(output->shape, output->ndim, false);
    for(uint64_t i=0; i<ones->size; i++) ones->data[i] = 1.0f;

    Tensor *one_minus_out = tensor_sub(ones, output);
    tensor_free(ones);

    Tensor *d_sigmoid = tensor_mul(output, one_minus_out);
    tensor_free(one_minus_out);

    Tensor *grad_input = tensor_mul(grad_output, d_sigmoid);
    tensor_free(d_sigmoid);

    return grad_input;
}

Tensor *tensor_tanh_backward(const Tensor *grad_output, const Tensor *output) {
    assert(grad_output != NULL);
    assert(output != NULL);

    // grad = grad_out * (1 - out^2)
    Tensor *output_sq = tensor_mul(output, output);

    Tensor *ones = tensor_zeros(output->shape, output->ndim, false);
    for(uint64_t i=0; i<ones->size; i++) ones->data[i] = 1.0f;

    Tensor *term = tensor_sub(ones, output_sq);
    tensor_free(ones);
    tensor_free(output_sq);

    Tensor *grad_input = tensor_mul(grad_output, term);
    tensor_free(term);
    return grad_input;
}

Tensor *tensor_gelu_backward(const Tensor *grad_output, const Tensor *input) {
    assert(grad_output != NULL);
    assert(input != NULL);

    Tensor *grad_input = tensor_create(NULL, grad_output->shape, grad_output->ndim, false);
    float32_t sqrt_2_over_pi = sqrtf(2.0f / (float32_t)M_PI);
    float32_t coeff = 0.044715f;

    for (uint64_t i = 0; i < input->size; i++) {
        float32_t x = input->data[i];
        float32_t x2 = x * x;
        float32_t x3 = x2 * x;

        float32_t tanh_arg = sqrt_2_over_pi * (x + coeff * x3);
        float32_t tanh_out = tanhf(tanh_arg);
        float32_t sech_sq = 1.0f - tanh_out * tanh_out;
        float32_t d_tanh_arg = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x2);

        float32_t gelu_grad = 0.5f * (1.0f + tanh_out) + 0.5f * x * sech_sq * d_tanh_arg;
        grad_input->data[i] = grad_output->data[i] * gelu_grad;
    }
    return grad_input;
}

Tensor *tensor_softmax_backward(const Tensor *grad_output, const Tensor *output, int64_t dim) {
    assert(grad_output != NULL);
    assert(output != NULL);

    // grad_input = output * (grad_output - sum(output * grad_output))
    Tensor *prod = tensor_mul(grad_output, output);
    Tensor *sum_prod = tensor_sum(prod, dim, true);
    tensor_free(prod);

    Tensor *sub = tensor_sub(grad_output, sum_prod);
    tensor_free(sum_prod);

    Tensor *grad_input = tensor_mul(output, sub);
    tensor_free(sub);
    return grad_input;
}

//
// Shape manipulation
//

Tensor *tensor_reshape_backward(const Tensor *grad_output, const uint64_t *old_shape, uint64_t old_ndim) {
    assert(grad_output != NULL);
    assert(old_shape != NULL);

    int64_t signed_shape[MAX_NDIM] = {0};
    for(uint64_t i=0; i<old_ndim; i++) signed_shape[i] = (int64_t)old_shape[i];

    return tensor_reshape(grad_output, signed_shape, old_ndim);
}

Tensor *tensor_transpose_backward(const Tensor *grad_output, uint64_t dim0, uint64_t dim1) {
    assert(grad_output != NULL);
    return tensor_transpose(grad_output, dim0, dim1);
}

Tensor *tensor_getitem_backward(const Tensor *grad_output, const Tensor *input, const uint64_t *multidim) {
    assert(grad_output != NULL);
    assert(input != NULL);
    assert(multidim != NULL);

    Tensor *grad_input = tensor_zeros(input->shape, input->ndim, false);

    if (input->strides) {
        uint64_t offset = 0;
        for (uint64_t i = 0; i < input->ndim; i++) {
            offset += multidim[i] * input->strides[i];
        }
        assert(offset < grad_input->size);
        // Assuming scalar grad_output for getitem
        grad_input->data[offset] = grad_output->data[0];
    }
    return grad_input;
}

//
// Losses
//

Tensor *tensor_mse_backward(const Tensor *grad_output, const Tensor *predictions, const Tensor *targets) {
    assert(grad_output != NULL);
    assert(predictions != NULL);
    assert(targets != NULL);
    assert(grad_output->size == 1);

    float32_t grad_scalar = grad_output->data[0];
    Tensor *grad_pred = tensor_create(NULL, predictions->shape, predictions->ndim, false);
    float32_t N = (float32_t)predictions->size;

    for (uint64_t i = 0; i < predictions->size; i++) {
        float32_t diff = predictions->data[i] - targets->data[i];
        grad_pred->data[i] = grad_scalar * 2.0f * diff / N;
    }
    return grad_pred;
}

Tensor *tensor_bce_backward(const Tensor *grad_output, const Tensor *predictions, const Tensor *targets) {
    assert(grad_output != NULL);
    assert(predictions != NULL);
    assert(targets != NULL);
    assert(grad_output->size == 1);

    float32_t grad_scalar = grad_output->data[0];
    Tensor *grad_pred = tensor_create(NULL, predictions->shape, predictions->ndim, false);
    float32_t N = (float32_t)predictions->size;

    for (uint64_t i = 0; i < predictions->size; i++) {
        float32_t pred = predictions->data[i];
        float32_t target = targets->data[i];
        if (pred < EPSILON) pred = EPSILON;
        if (pred > 1.0f - EPSILON) pred = 1.0f - EPSILON;

        float32_t denom = pred * (1.0f - pred) * N;
        grad_pred->data[i] = grad_scalar * (pred - target) / denom;
    }
    return grad_pred;
}

static inline float32_t compute_softmax_prob(const float32_t *logits, uint64_t offset, float32_t max_val, float32_t sum_exp, uint64_t class_idx) { return expf(logits[offset + class_idx] - max_val) / sum_exp; }

Tensor *tensor_crossentropy_backward(const Tensor *grad_output, const Tensor *logits, const Tensor *targets) {
    assert(grad_output != NULL);
    assert(logits != NULL);
    assert(targets != NULL);
    assert(grad_output->size == 1);

    float32_t grad_scalar = grad_output->data[0];
    uint64_t batch_size = logits->shape[0];
    uint64_t class_count = logits->shape[1];
    float32_t batch_size_f32 = (float32_t)batch_size;

    Tensor *grad_logits = tensor_create(NULL, logits->shape, logits->ndim, false);

    for (uint64_t i = 0; i < batch_size; i++) {
        uint64_t offset = i * class_count;
        float32_t max_logit = -INFINITY;
        for (uint64_t c = 0; c < class_count; c++) {
            if (logits->data[offset + c] > max_logit) max_logit = logits->data[offset + c];
        }
        float32_t exp_sum = 0.0f;
        for (uint64_t c = 0; c < class_count; c++) {
            exp_sum += expf(logits->data[offset + c] - max_logit);
        }
        uint64_t target = (uint64_t)targets->data[i];

        for (uint64_t c = 0; c < class_count; c++) {
            float32_t prob = compute_softmax_prob(logits->data, offset, max_logit, exp_sum, c);
            float32_t one_hot = (c == target) ? 1.0f : 0.0f;
            grad_logits->data[offset + c] = grad_scalar * (prob - one_hot) / batch_size_f32;
        }
    }
    return grad_logits;
}
