#pragma once

#include "tensor.h"

Tensor *tensor_sigmoid_backward(const Tensor *t);
// todo: autograd backward

Tensor *tensor_relu_backward(const Tensor *t);
// todo: autograd backward

Tensor *tensor_tanh_backward(const Tensor *t);
// todo: autograd backward

Tensor *tensor_gelu_backward(const Tensor *t);
// todo: autograd backward

Tensor *tensor_softmax_backward(const Tensor *t, int64_t dim);
// todo: autograd backward
