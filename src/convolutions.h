#pragma once

#include "layers.h"
#include "tensor.h"

/**
 * creates a 2D convolution layer.
 *
 * @param in_channels  number of input channels
 * @param out_channels number of output channels
 * @param kernel_size  size of the convolving kernel
 * @param stride       stride of the convolution (default: 1)
 * @param padding      zero-padding added to both sides of the input (default: 0)
 * @param bias         if true, adds a learnable bias to the output
 * @return             pointer to new layer
 */
Layer *layer_conv2d_create(uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size, uint64_t stride, uint64_t padding, bool bias);

/**
 * computes gradients for conv2d backward pass.
 *
 * @param input         input tensor from forward pass
 * @param weight        weight tensor
 * @param bias          bias tensor (can be NULL)
 * @param stride        stride used in forward pass
 * @param padding       padding used in forward pass
 * @param kernel_size   kernel size used in forward pass
 * @param grad_output   gradient flowing back from next layer
 * @param out_grad_in   output gradient w.r.t input (newly allocated)
 * @param out_grad_w    output gradient w.r.t weight (newly allocated)
 * @param out_grad_b    output gradient w.r.t bias (newly allocated, can be NULL)
 */
void conv2d_backward(const Tensor *input, const Tensor *weight, const Tensor *bias, uint64_t stride, uint64_t padding, uint64_t kernel_size, const Tensor *grad_output, Tensor **out_grad_in, Tensor **out_grad_w, Tensor **out_grad_b);

/**
 * creates a 2D max pooling layer.
 *
 * @param kernel_size  size of the pooling region
 * @param stride       stride of the pooling operation (default: kernel_size)
 * @param padding      zero-padding added to both sides of the input (default: 0)
 * @return             pointer to new layer
 */
Layer *layer_maxpool2d_create(uint64_t kernel_size, uint64_t stride, uint64_t padding);

/**
 * computes gradients for maxpool2d backward pass.
 *
 * @param input         input tensor from forward pass
 * @param output_shape  shape of the output tensor from forward pass (array of 4 uint64_t)
 * @param kernel_size   kernel size
 * @param stride        stride
 * @param padding       padding
 * @param grad_output   gradient from next layer
 * @return              gradient w.r.t input (newly allocated)
 */
Tensor *maxpool2d_backward(const Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding, const Tensor *grad_output);

/**
 * creates a 2D average pooling layer.
 *
 * @param kernel_size  size of the pooling region
 * @param stride       stride of the pooling operation (default: kernel_size)
 * @param padding      zero-padding added to both sides of the input (default: 0)
 * @return             pointer to new layer
 */
Layer *layer_avgpool2d_create(uint64_t kernel_size, uint64_t stride, uint64_t padding);

/**
 * creates a batch normalization layer for use with 2D inputs.
 *
 * @param num_features number of features (channels)
 * @param eps          epsilon for stability (default: 1e-5)
 * @param momentum     momentum for running stats (default: 0.1)
 * @return             pointer to new layer
 */
Layer *layer_batchnorm2d_create(uint64_t num_features, float32_t eps, float32_t momentum);

/**
 * creates a simple CNN model.
 *
 * architecture:
 *   conv2d(3->16, 3x3) -> relu -> maxpool(2x2)
 *   conv2d(16->32, 3x3) -> relu -> maxpool(2x2)
 *   flatten -> linear(features->num_classes)
 *
 * @param num_classes  number of output classes
 * @return             pointer to new layer
 */
Layer *simple_cnn_create(uint64_t num_classes);
