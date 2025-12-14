#pragma once

#include "tensor.h"
#include <stdint.h>

Tensor *tensor_sigmoid(Tensor *t);
Tensor *tensor_relu(Tensor *t);
Tensor *tensor_tanh(Tensor *t);
Tensor *tensor_gelu(Tensor *t);
Tensor *tensor_softmax(Tensor *t, int64_t dim);
