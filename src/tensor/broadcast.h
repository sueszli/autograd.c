#pragma once

#include "../utils/types.h"
#include "tensor.h"
#include <stdbool.h>

typedef struct {
    i32 *shape;
    i32 ndim;
} shape_t;

void shape_free(shape_t *s);

bool tensor_can_broadcast(const tensor_t *a, const tensor_t *b);

shape_t get_tensor_broadcast_shape(const tensor_t *a, const tensor_t *b);

tensor_t *tensor_broadcast_to(const tensor_t *tensor, const i32 *target_shape, i32 target_ndim);
