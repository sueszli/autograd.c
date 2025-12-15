#pragma once

#include "layers.h"
#include "tensor.h"

void conv2d_backward(const Tensor *input, const Tensor *weight, const Tensor *bias, uint64_t stride, uint64_t padding, uint64_t kernel_size, const Tensor *grad_output, Tensor **out_grad_in, Tensor **out_grad_w, Tensor **out_grad_b);
// todo: conv2d_backward_fn

Tensor *maxpool2d_backward(const Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding, const Tensor *grad_output);
// todo: maxpool2d_backward_fn

Tensor *avgpool2d_backward(const Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding, const Tensor *grad_output);
// todo: avgpool2d_backward_fn

void batchnorm2d_backward(const Tensor *input, const Tensor *gamma, const Tensor *batch_mean, const Tensor *batch_var, float32_t eps, const Tensor *grad_output, Tensor **out_grad_in, Tensor **out_grad_gamma, Tensor **out_grad_beta);
// todo: batchnorm2d_backward_fn
