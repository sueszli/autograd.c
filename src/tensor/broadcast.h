#pragma once

#include "../utils/types.h"
#include "tensor.h"
#include <stdbool.h>

typedef struct {
    tensor_t *a;
    tensor_t *b;
} broadcasted_tensors_t;

broadcasted_tensors_t tensor_broadcast(tensor_t *a, tensor_t *b);

bool tensor_shapes_match(const tensor_t *a, const tensor_t *b);

tensor_t *tensor_reduce(const tensor_t *broadcasted_grad, const tensor_t *target_tensor);
