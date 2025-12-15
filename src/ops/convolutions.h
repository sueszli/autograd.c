#pragma once

#include "tensor.h"

Tensor *tensor_conv2d(const Tensor *input, const Tensor *weight, const Tensor *bias, uint64_t stride, uint64_t padding, uint64_t dilation);
Tensor *tensor_maxpool2d(const Tensor *input, uint64_t kernel_size, uint64_t stride, uint64_t padding);
Tensor *tensor_avgpool2d(const Tensor *input, uint64_t kernel_size, uint64_t stride, uint64_t padding);
Tensor *tensor_batchnorm2d(const Tensor *input, const Tensor *gamma, const Tensor *beta, Tensor *running_mean, Tensor *running_var, bool training, float32_t momentum, float32_t eps);
