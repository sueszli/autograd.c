#pragma once

#include "tensor.h"
#include <stdint.h>

Tensor *tensor_reshape(const Tensor *t, const int64_t *new_shape, uint64_t new_ndim);
Tensor *tensor_transpose(const Tensor *t, uint64_t dim0, uint64_t dim1);
