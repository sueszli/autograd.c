#pragma once

#include "tensor.h"

// can't be named just `tanh` due to conflict with standard library
Tensor *tensor_sigmoid(const Tensor *t);
Tensor *tensor_relu(const Tensor *t);
Tensor *tensor_tanh(const Tensor *t);
Tensor *tensor_gelu(const Tensor *t);
Tensor *tensor_softmax(const Tensor *t, int64_t dim);
