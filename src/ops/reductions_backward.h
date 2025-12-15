#pragma once

#include "tensor.h"
#include <stdbool.h>

Tensor *tensor_sum_backward(const Tensor *grad_output, const Tensor *t, int64_t dim_idx, bool keepdims);
// todo: autograd backward

Tensor *tensor_mean_backward(const Tensor *grad_output, const Tensor *t, int64_t dim_idx, bool keepdims);
// todo: autograd backward

Tensor *tensor_max_backward(const Tensor *grad_output, const Tensor *t, const Tensor *out, int64_t dim_idx, bool keepdims);
// todo: autograd backward
