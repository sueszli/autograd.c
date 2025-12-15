#pragma once

#include "tensor.h"
#include <stdbool.h>
#include <stdint.h>

//
// backward math functions
//

// Arithmetic
void tensor_add_backward(const Tensor *grad_output, const Tensor *a, const Tensor *b, Tensor **out_grad_a, Tensor **out_grad_b);
void tensor_sub_backward(const Tensor *grad_output, const Tensor *a, const Tensor *b, Tensor **out_grad_a, Tensor **out_grad_b);
void tensor_mul_backward(const Tensor *grad_output, const Tensor *a, const Tensor *b, Tensor **out_grad_a, Tensor **out_grad_b);
void tensor_div_backward(const Tensor *grad_output, const Tensor *a, const Tensor *b, Tensor **out_grad_a, Tensor **out_grad_b);
void tensor_matmul_backward(const Tensor *grad_output, const Tensor *a, const Tensor *b, Tensor **out_grad_a, Tensor **out_grad_b);

// Reductions
Tensor *tensor_sum_backward(const Tensor *grad_output, const Tensor *input, int64_t dim_idx, bool keepdims);
Tensor *tensor_mean_backward(const Tensor *grad_output, const Tensor *input, int64_t dim_idx, bool keepdims);
Tensor *tensor_max_backward(const Tensor *grad_output, const Tensor *input, const Tensor *output, int64_t dim_idx, bool keepdims);

// Activations
Tensor *tensor_relu_backward(const Tensor *grad_output, const Tensor *input);
Tensor *tensor_sigmoid_backward(const Tensor *grad_output, const Tensor *output);
Tensor *tensor_tanh_backward(const Tensor *grad_output, const Tensor *output);
Tensor *tensor_gelu_backward(const Tensor *grad_output, const Tensor *input);
Tensor *tensor_softmax_backward(const Tensor *grad_output, const Tensor *output, int64_t dim);

// Shape manipulation
Tensor *tensor_reshape_backward(const Tensor *grad_output, const uint64_t *old_shape, uint64_t old_ndim);
Tensor *tensor_transpose_backward(const Tensor *grad_output, uint64_t dim0, uint64_t dim1);
Tensor *tensor_getitem_backward(const Tensor *grad_output, const Tensor *input, const uint64_t *multidim);

// Losses
Tensor *tensor_mse_backward(const Tensor *grad_output, const Tensor *predictions, const Tensor *targets);
Tensor *tensor_bce_backward(const Tensor *grad_output, const Tensor *predictions, const Tensor *targets);
Tensor *tensor_crossentropy_backward(const Tensor *grad_output, const Tensor *logits, const Tensor *targets);
