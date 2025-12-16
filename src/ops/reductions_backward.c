#include "ops/reductions_backward.h"
#include "ops/arithmetic.h"
#include "ops/reductions.h"
#include "ops/reshapes.h"
#include "tensor.h"
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

//
// sum
//

Tensor *tensor_sum_backward(const Tensor *grad_output, const Tensor *t, int64_t dim_idx, bool keepdims) {
    assert(grad_output != NULL);
    assert(t != NULL);

    int64_t ndim_signed = (int64_t)t->ndim;
    int64_t target_dim_signed = (dim_idx < 0) ? (dim_idx + ndim_signed) : dim_idx;
    assert(target_dim_signed >= 0 && target_dim_signed < ndim_signed && "target_dim out of bounds");

    uint64_t target_dim = (uint64_t)target_dim_signed;

    const Tensor *grad_expanded = grad_output;
    bool needs_free = false;

    if (!keepdims) {
        assert(t->ndim <= MAX_NDIM);
        int64_t new_shape[MAX_NDIM] = {0};
        uint64_t grad_dim_idx = 0;

        for (uint64_t d = 0; d < t->ndim; d++) {
            if (d == target_dim) {
                new_shape[d] = 1;
            } else {
                new_shape[d] = (int64_t)grad_output->shape[grad_dim_idx++];
            }
        }

        grad_expanded = tensor_reshape(grad_output, new_shape, t->ndim);
        needs_free = true;
    }

    Tensor *zeros = tensor_zeros(t->shape, t->ndim, false);
    assert(zeros != NULL);

    Tensor *grad_input = tensor_add(zeros, grad_expanded);
    assert(grad_input != NULL);

    tensor_free(zeros);
    if (needs_free) {
        tensor_free((Tensor *)grad_expanded);
    }

    return grad_input;
}

void sum_backward(Function *fn, const Tensor *grad_output) {
    assert(fn != NULL);
    assert(grad_output != NULL);
    assert(fn->num_inputs == 1);
    assert(fn->ctx != NULL && "sum_backward requires context");

    Tensor *t = fn->inputs[0];
    const SumContext *ctx = (SumContext *)fn->ctx;

    if (t != NULL && t->requires_grad) {
        Tensor *grad_t = tensor_sum_backward(grad_output, t, ctx->dim_idx, ctx->keepdims);
        accumulate_grad(t, grad_t);
    }

    free(fn->ctx);
    fn->ctx = NULL;
}

//
// mean
//

Tensor *tensor_mean_backward(const Tensor *grad_output, const Tensor *t, int64_t dim_idx, bool keepdims) {
    assert(grad_output != NULL);
    assert(t != NULL);

    Tensor *sum_grad = tensor_sum_backward(grad_output, t, dim_idx, keepdims);
    assert(sum_grad != NULL);

    int64_t ndim_signed = (int64_t)t->ndim;
    int64_t target_dim_signed = (dim_idx < 0) ? (dim_idx + ndim_signed) : dim_idx;
    assert(target_dim_signed >= 0 && target_dim_signed < ndim_signed && "target_dim out of bounds");
    uint64_t target_dim = (uint64_t)target_dim_signed;

    uint64_t count = t->shape[target_dim];

    if (count == 0) {
        return sum_grad;
    }

    const uint64_t scalar_shape[] = {1};
    Tensor *scale_t = tensor_create(NULL, scalar_shape, 1, false);
    assert(scale_t != NULL);

    scale_t->data[0] = 1.0f / (float32_t)count;

    Tensor *grad_input = tensor_mul(sum_grad, scale_t, true); // disable_grad=true
    assert(grad_input != NULL);

    tensor_free(sum_grad);
    tensor_free(scale_t);

    return grad_input;
}

void mean_backward(Function *fn, const Tensor *grad_output) {
    assert(fn != NULL);
    assert(grad_output != NULL);
    assert(fn->num_inputs == 1);
    assert(fn->ctx != NULL && "mean_backward requires context");

    Tensor *t = fn->inputs[0];
    const MeanContext *ctx = (MeanContext *)fn->ctx;

    if (t != NULL && t->requires_grad) {
        Tensor *grad_t = tensor_mean_backward(grad_output, t, ctx->dim_idx, ctx->keepdims);
        accumulate_grad(t, grad_t);
    }

    free(fn->ctx);
    fn->ctx = NULL;
}

//
// max
//

Tensor *tensor_max_backward(const Tensor *grad_output, const Tensor *t, const Tensor *out, int64_t dim_idx, bool keepdims) {
    assert(grad_output != NULL);
    assert(t != NULL);
    assert(out != NULL);

    int64_t ndim_signed = (int64_t)t->ndim;
    int64_t target_dim_signed = (dim_idx < 0) ? (dim_idx + ndim_signed) : dim_idx;
    assert(target_dim_signed >= 0 && target_dim_signed < ndim_signed && "target_dim out of bounds");
    uint64_t target_dim = (uint64_t)target_dim_signed;

    Tensor *grad_input = tensor_zeros(t->shape, t->ndim, false);
    assert(grad_input != NULL);

    uint64_t curr_coords[MAX_NDIM] = {0};
    uint64_t out_coords[MAX_NDIM] = {0};

    for (uint64_t i = 0; i < t->size; i++) {
        linear_to_multidim_mut(i, t->shape, t->ndim, curr_coords);

        uint64_t out_offset = 0;
        if (keepdims) {
            uint64_t saved_dim_val = curr_coords[target_dim];
            // for keepdims=true, target dimension is collapsed to size 1 (index 0)
            curr_coords[target_dim] = 0;
            out_offset = multidim_to_linear(curr_coords, t->ndim, out->shape, out->ndim, out->strides);
            curr_coords[target_dim] = saved_dim_val;
        } else {
            uint64_t k = 0;
            for (uint64_t d = 0; d < t->ndim; d++) {
                if (d != target_dim) {
                    out_coords[k++] = curr_coords[d];
                }
            }
            out_offset = multidim_to_linear(out_coords, out->ndim, out->shape, out->ndim, out->strides);
        }

        float32_t val = t->data[i];
        float32_t max_val = out->data[out_offset];

        if (val == max_val) {
            grad_input->data[i] += grad_output->data[out_offset];
        }
    }

    return grad_input;
}

void max_backward(Function *fn, const Tensor *grad_output) {
    assert(fn != NULL);
    assert(grad_output != NULL);
    assert(fn->num_inputs == 1);
    assert(fn->ctx != NULL && "max_backward requires context");

    Tensor *t = fn->inputs[0];
    const MaxContext *ctx = (MaxContext *)fn->ctx;

    if (t != NULL && t->requires_grad) {
        Tensor *grad_t = tensor_max_backward(grad_output, t, ctx->output, ctx->dim_idx, ctx->keepdims);
        accumulate_grad(t, grad_t);
    }

    free(fn->ctx);
    fn->ctx = NULL;
}
