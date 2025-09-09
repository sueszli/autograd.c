#pragma once

#include "../utils/types.h"
#include "tensor.h"
#include <stdbool.h>

typedef enum {
    TENSOR_OP_ADD,
    TENSOR_OP_SUB,
    TENSOR_OP_MUL,
    TENSOR_OP_DIV
} tensor_op_t;

tensor_t *tensor_op_add(tensor_t *a, tensor_t *b, bool use_broadcasting);
tensor_t *tensor_op_sub(tensor_t *a, tensor_t *b, bool use_broadcasting);
tensor_t *tensor_op_mul(tensor_t *a, tensor_t *b, bool use_broadcasting);
tensor_t *tensor_op_div(tensor_t *a, tensor_t *b, bool use_broadcasting);

tensor_t *tensor_op_generic(tensor_t *a, tensor_t *b, tensor_op_t op, bool use_broadcasting);