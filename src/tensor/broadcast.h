#pragma once

#include "../utils/types.h"
#include "tensor.h"
#include <stdbool.h>

bool tensor_can_broadcast(const Tensor *a, const Tensor *b);

i32 *get_tensor_broadcast_shape(const Tensor *a, const Tensor *b, i32 *result_ndim);

Tensor *tensor_broadcast_to(const Tensor *tensor, const i32 *target_shape, i32 target_ndim);
