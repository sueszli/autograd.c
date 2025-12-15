#pragma once

#include "tensor.h"

Tensor *tensor_sigmoid_backward(const Tensor *t);
Tensor *tensor_relu_backward(const Tensor *t);
Tensor *tensor_tanh_backward(const Tensor *t);
Tensor *tensor_gelu_backward(const Tensor *t);
Tensor *tensor_softmax_backward(const Tensor *t, int64_t dim);
