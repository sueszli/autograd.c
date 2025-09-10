#pragma once

#include "../utils/types.h"
#include "tensor.h"
#include <stdbool.h>

typedef struct {
    tensor_t *a;
    tensor_t *b;
} broadcasted_tensors_t;

broadcasted_tensors_t tensor_broadcast(tensor_t *a, tensor_t *b);
