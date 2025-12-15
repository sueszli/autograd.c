#pragma once

#include "tensor.h"
#include <stdbool.h>
#include <stdint.h>

Tensor *tensor_sum(const Tensor *t, int64_t dim_idx, bool keepdims);
Tensor *tensor_mean(const Tensor *t, int64_t dim_idx, bool keepdims);
Tensor *tensor_max(const Tensor *t, int64_t dim_idx, bool keepdims);
