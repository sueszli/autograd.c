#include "ops/arithmetic_backward.h"
#include "ops/arithmetic.h"
#include "tensor.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>

static Tensor *unbroadcast(const Tensor *grad, const Tensor *input) {
    assert(grad != NULL);
    assert(input != NULL);

    const Tensor *curr_grad = grad;
    bool owns_tensor = false;

    // broadcasting adds dimensions on the left, so collapse extra leading dimensions
    // e.g., grad (2, 3, 4) -> input (3, 4) => sum dim 0
    while (curr_grad->ndim > input->ndim) {
        Tensor *summed = tensor_sum(curr_grad, 0, false);
        if (owns_tensor) {
            tensor_free((Tensor *)curr_grad);
        }
        curr_grad = summed;
        owns_tensor = true;
    }

    assert(curr_grad->ndim == input->ndim);
    assert(input->ndim < MAX_NDIM);

    // dimensions match. collapse any dims where input had size 1 (broadcasted dim)
    // e.g., grad (3, 4) -> input (1, 4) => sum dim 0
    for (uint64_t dim_idx = 0; dim_idx < input->ndim; dim_idx++) {
        if (input->shape[dim_idx] == 1 && curr_grad->shape[dim_idx] > 1) {
            Tensor *summed = tensor_sum(curr_grad, (int64_t)dim_idx, true);
            if (owns_tensor) {
                tensor_free((Tensor *)curr_grad);
            }
            curr_grad = summed;
            owns_tensor = true;
        }
    }

    // if no reduction happened, ensure we return a new tensor to respect ownership contract
    if (!owns_tensor) {
        return tensor_create(grad->data, grad->shape, grad->ndim, false);
    }

    return (Tensor *)curr_grad;
}

Tensor *tensor_add_backward_a(const Tensor *grad_output, const Tensor *a) {
    assert(grad_output != NULL);
    assert(a != NULL);
    return unbroadcast(grad_output, a);
}

Tensor *tensor_add_backward_b(const Tensor *grad_output, const Tensor *b) {
    assert(grad_output != NULL);
    assert(b != NULL);
    return unbroadcast(grad_output, b);
}

Tensor *tensor_sub_backward_a(const Tensor *grad_output, const Tensor *a) {
    assert(grad_output != NULL);
    assert(a != NULL);
    return unbroadcast(grad_output, a);
}

Tensor *tensor_sub_backward_b(const Tensor *grad_output, const Tensor *b) {
    assert(grad_output != NULL);
    assert(b != NULL);

    Tensor *zeros = tensor_zeros(grad_output->shape, grad_output->ndim, false);
    Tensor *neg_grad = tensor_sub(zeros, grad_output);
    tensor_free(zeros);

    Tensor *result = unbroadcast(neg_grad, b);
    tensor_free(neg_grad);
    return result;
}

Tensor *tensor_mul_backward_a(const Tensor *grad_output, const Tensor *a, const Tensor *b) {
    assert(grad_output != NULL);
    assert(a != NULL);
    assert(b != NULL);

    Tensor *temp = tensor_mul(grad_output, b);
    Tensor *result = unbroadcast(temp, a);
    tensor_free(temp);
    return result;
}

Tensor *tensor_mul_backward_b(const Tensor *grad_output, const Tensor *a, const Tensor *b) {
    assert(grad_output != NULL);
    assert(a != NULL);
    assert(b != NULL);

    Tensor *temp = tensor_mul(grad_output, a);
    Tensor *result = unbroadcast(temp, b);
    tensor_free(temp);
    return result;
}

Tensor *tensor_div_backward_a(const Tensor *grad_output, const Tensor *a, const Tensor *b) {
    assert(grad_output != NULL);
    assert(a != NULL);
    assert(b != NULL);

    Tensor *temp = tensor_div(grad_output, b);
    Tensor *result = unbroadcast(temp, a);
    tensor_free(temp);
    return result;
}

Tensor *tensor_div_backward_b(const Tensor *grad_output, const Tensor *a, const Tensor *b) {
    assert(grad_output != NULL);
    assert(a != NULL);
    assert(b != NULL);

    Tensor *zeros = tensor_zeros(grad_output->shape, grad_output->ndim, false);
    Tensor *neg_grad = tensor_sub(zeros, grad_output);
    tensor_free(zeros);

    Tensor *num = tensor_mul(neg_grad, a);
    tensor_free(neg_grad);

    Tensor *b_sq = tensor_mul(b, b);

    Tensor *temp = tensor_div(num, b_sq);
    tensor_free(num);
    tensor_free(b_sq);

    Tensor *result = unbroadcast(temp, b);
    tensor_free(temp);
    return result;
}

Tensor *tensor_matmul_backward_a(const Tensor *grad_output, const Tensor *a, const Tensor *b) {
    assert(grad_output != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(b->ndim >= 2);

    Tensor *b_T = tensor_transpose(b, b->ndim - 2, b->ndim - 1);
    Tensor *temp = tensor_matmul(grad_output, b_T);
    tensor_free(b_T);

    Tensor *result = unbroadcast(temp, a);
    tensor_free(temp);
    return result;
}

Tensor *tensor_matmul_backward_b(const Tensor *grad_output, const Tensor *a, const Tensor *b) {
    assert(grad_output != NULL);
    assert(a != NULL);
    assert(b != NULL);
    assert(a->ndim >= 2);

    Tensor *a_T = tensor_transpose(a, a->ndim - 2, a->ndim - 1);
    Tensor *temp = tensor_matmul(a_T, grad_output);
    tensor_free(a_T);

    Tensor *result = unbroadcast(temp, b);
    tensor_free(temp);
    return result;
}
