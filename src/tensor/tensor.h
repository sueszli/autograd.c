#pragma once

#include "../utils/types.h"
#include <stdbool.h>
#include <stddef.h>

typedef struct tensor_t tensor_t;

typedef void (*grad_fn)(tensor_t *); // for gradient functions

struct tensor_t {
    f32 *data;
    i32 *shape;
    i32 ndim;
    bool requires_grad;
    tensor_t *grad;
    grad_fn grad_fn;
    void **ctx; // context for backward pass (e.g., saved tensors)
    i32 ctx_size;
};

typedef struct {
    tensor_t *a;
    tensor_t *b;
} tensor_pair_t;

tensor_t *tensor_create(f32 *data, i32 *shape, i32 ndim, bool requires_grad);
void tensor_destroy(tensor_t *t);

void tensor_print(const tensor_t *t);
void tensor_zero_grad(tensor_t *t);
u64 tensor_size(const tensor_t *t);

tensor_t *tensor_transpose(tensor_t *a);
