#pragma once

#include "tensor.h"

tensor_t *tensor_add(tensor_t *a, tensor_t *b);
tensor_t *tensor_sub(tensor_t *a, tensor_t *b);
tensor_t *tensor_mul(tensor_t *a, tensor_t *b);
tensor_t *tensor_div(tensor_t *a, tensor_t *b);
tensor_t *tensor_matmul(tensor_t *a, tensor_t *b);
tensor_t *tensor_relu(tensor_t *a);
tensor_t *tensor_softmax(tensor_t *a);
tensor_t *tensor_cross_entropy(tensor_t *a, i32 target_idx);

void cross_entropy_backward(tensor_t *t);

tensor_t *tensor_conv2d(tensor_t *input, tensor_t *kernel, i32 stride, i32 padding);
tensor_t *tensor_max_pool2d(tensor_t *input, i32 kernel_size, i32 stride);
tensor_t *tensor_avg_pool2d(tensor_t *input, i32 kernel_size, i32 stride);
