#pragma once

#include "autograd.h"
#include <stdint.h>

struct Tensor;

//
// backward constructors
//

struct GradFn *new_add_backward(struct Tensor *a, struct Tensor *b);
struct GradFn *new_sub_backward(struct Tensor *a, struct Tensor *b);
struct GradFn *new_mul_backward(struct Tensor *a, struct Tensor *b);
struct GradFn *new_div_backward(struct Tensor *a, struct Tensor *b);
struct GradFn *new_sum_backward(struct Tensor *input, int64_t dim_idx, bool keepdims);
struct GradFn *new_matmul_backward(struct Tensor *a, struct Tensor *b);
struct GradFn *new_relu_backward(struct Tensor *input);
struct GradFn *new_sigmoid_backward(struct Tensor *input, struct Tensor *output);
struct GradFn *new_softmax_backward(struct Tensor *input, struct Tensor *output, int64_t dim);
struct GradFn *new_reshape_backward(struct Tensor *input, const uint64_t *old_shape, uint64_t old_ndim);
struct GradFn *new_transpose_backward(struct Tensor *input, uint64_t dim0, uint64_t dim1);
struct GradFn *new_getitem_backward(struct Tensor *input, const uint64_t *multidim);
struct GradFn *new_gelu_backward(struct Tensor *input);
struct GradFn *new_mse_backward(struct Tensor *predictions, struct Tensor *targets);
struct GradFn *new_bce_backward(struct Tensor *predictions, struct Tensor *targets);
struct GradFn *new_crossentropy_backward(struct Tensor *logits, struct Tensor *targets);
struct GradFn *new_tanh_backward(struct Tensor *input, struct Tensor *output);
struct GradFn *new_mean_backward(struct Tensor *input, int64_t dim_idx, bool keepdims);
struct GradFn *new_max_backward(struct Tensor *input, struct Tensor *output, int64_t dim_idx, bool keepdims);
struct GradFn *new_conv2d_backward(struct Tensor *input, struct Tensor *weight, struct Tensor *bias, uint64_t stride, uint64_t padding, uint64_t kernel_size);
struct GradFn *new_maxpool2d_backward(struct Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding);
struct GradFn *new_avgpool2d_backward(struct Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding);
struct GradFn *new_batchnorm2d_backward(struct Tensor *input, struct Tensor *gamma, struct Tensor *batch_mean, struct Tensor *batch_var, float32_t eps);
