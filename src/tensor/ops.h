#pragma once

#include "tensor.h"

tensor_t *tensor_add(tensor_t *a, tensor_t *b);
tensor_t *tensor_sub(tensor_t *a, tensor_t *b);
tensor_t *tensor_mul(tensor_t *a, tensor_t *b);
tensor_t *tensor_div(tensor_t *a, tensor_t *b);