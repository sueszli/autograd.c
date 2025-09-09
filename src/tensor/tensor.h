#pragma once

#include "../utils/types.h"
#include <stdbool.h>
#include <stddef.h>

typedef struct Tensor Tensor;

typedef void (*grad_fn)(Tensor *); // for gradient functions

struct Tensor {
    f32 *data;
    i32 *shape;
    i32 ndim;
    bool requires_grad;
    Tensor *grad;
    grad_fn grad_fn;
    void **ctx; // context for backward pass (e.g., saved tensors)
    i32 ctx_size;
};

Tensor *tensor_create(f32 *data, i32 *shape, i32 ndim, bool requires_grad);
void tensor_destroy(Tensor *t);

void tensor_pri32(const Tensor *t);
void tensor_zero_grad(Tensor *t);
u64 tensor_size(const Tensor *t);
Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_mul(Tensor *a, Tensor *b);
Tensor *tensor_matmul(Tensor *a, Tensor *b);
Tensor *tensor_relu(Tensor *a);
Tensor *tensor_softmax(Tensor *a);
Tensor *tensor_cross_entropy(Tensor *a, i32 target_idx);
Tensor *tensor_transpose(Tensor *a);

void cross_entropy_backward(Tensor *t);
