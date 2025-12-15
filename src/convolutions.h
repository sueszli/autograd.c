#pragma once

#include "layers.h"
#include "tensor.h"

// 2d convolution layer
Layer *layer_conv2d_create(uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size, uint64_t stride, uint64_t padding, bool bias);

// computes gradients for conv2d backward pass
void conv2d_backward(const Tensor *input, const Tensor *weight, const Tensor *bias, uint64_t stride, uint64_t padding, uint64_t kernel_size, const Tensor *grad_output, Tensor **out_grad_in, Tensor **out_grad_w, Tensor **out_grad_b);

// 2d max pooling layer
Layer *layer_maxpool2d_create(uint64_t kernel_size, uint64_t stride, uint64_t padding);

// computes gradients for maxpool2d backward pass
Tensor *maxpool2d_backward(const Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding, const Tensor *grad_output);

// 2d average pooling layer
Layer *layer_avgpool2d_create(uint64_t kernel_size, uint64_t stride, uint64_t padding);

// computes gradients for avgpool2d backward pass
Tensor *avgpool2d_backward(const Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding, const Tensor *grad_output);

// batch normalization layer
Layer *layer_batchnorm2d_create(uint64_t num_features, float32_t eps, float32_t momentum);

// computes gradients for batchnorm2d backward pass
void batchnorm2d_backward(const Tensor *input, const Tensor *gamma, const Tensor *batch_mean, const Tensor *batch_var, float32_t eps, const Tensor *grad_output, Tensor **out_grad_in, Tensor **out_grad_gamma, Tensor **out_grad_beta);

// simple cnn model: conv->relu->pool -> conv->relu->pool -> flatten->linear
Layer *simple_cnn_create(uint64_t num_classes);
