#pragma once

#include "layers.h"
#include "tensor.h"

Layer *layer_conv2d_create(uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size, uint64_t stride, uint64_t padding, bool bias);
void conv2d_backward(const Tensor *input, const Tensor *weight, const Tensor *bias, uint64_t stride, uint64_t padding, uint64_t kernel_size, const Tensor *grad_output, Tensor **out_grad_in, Tensor **out_grad_w, Tensor **out_grad_b);

Layer *layer_maxpool2d_create(uint64_t kernel_size, uint64_t stride, uint64_t padding);
Tensor *maxpool2d_backward(const Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding, const Tensor *grad_output);

Layer *layer_avgpool2d_create(uint64_t kernel_size, uint64_t stride, uint64_t padding);
Tensor *avgpool2d_backward(const Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding, const Tensor *grad_output);

Layer *layer_batchnorm2d_create(uint64_t num_features, float32_t eps, float32_t momentum);
void batchnorm2d_backward(const Tensor *input, const Tensor *gamma, const Tensor *batch_mean, const Tensor *batch_var, float32_t eps, const Tensor *grad_output, Tensor **out_grad_in, Tensor **out_grad_gamma, Tensor **out_grad_beta);
