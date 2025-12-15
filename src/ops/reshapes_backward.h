#pragma once

#include "tensor.h"

Tensor *tensor_reshape_backward(const Tensor *grad_output, const Tensor *input);
Tensor *tensor_transpose_backward(const Tensor *grad_output, uint64_t dim0, uint64_t dim1);
