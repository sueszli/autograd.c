#pragma once

#include "layers.h"
#include "tensor.h"

Layer *layer_conv2d_create(uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size, uint64_t stride, uint64_t padding, bool bias);
Layer *layer_maxpool2d_create(uint64_t kernel_size, uint64_t stride, uint64_t padding);
Layer *layer_avgpool2d_create(uint64_t kernel_size, uint64_t stride, uint64_t padding);
Layer *layer_batchnorm2d_create(uint64_t num_features, float32_t eps, float32_t momentum);
