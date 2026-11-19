#pragma once

#include "std/types.h"

static constexpr i64 MAX_NDIM = 4;

//
// n-d row-major f64 tensors, named after pytorch.
// https://pytorch.org/docs/stable/tensors.html
//
// data is contiguous, there are no strides and no broadcasting.
// params and inputs live in the perm arena, which is never reset.
// op outputs live in the graph arena, see autograd.h.
//

typedef struct Tensor {
    f64 *data;
    f64 *grad;         // params get one in tensor_new, op outputs when backward reaches them
    struct Node *node; // the op that made this tensor, NULL for params and inputs
    i64 shape[MAX_NDIM];
    i64 ndim; // 0 for a scalar
    i64 size; // number of elements
    bool requires_grad;
} Tensor;

// a shape literal and its length, eg. tensor_new(shape(2, 3), NULL, true)
#define shape(...) (const i64[]){__VA_ARGS__}, (i64)(sizeof((i64[]){__VA_ARGS__}) / sizeof(i64))

// in perm, data NULL gives zeros
Tensor *tensor_new(const i64 *shape, i64 ndim, const f64 *data, bool requires_grad);

// copies the shape and multiplies it out into size
void tensor_set_shape(Tensor *t, const i64 *shape, i64 ndim);

// the last dim, which row-wise ops act on
static inline i64 cols_of(const Tensor *t) { return t->shape[t->ndim - 1]; }

// every leading dim flattened into rows
static inline i64 rows_of(const Tensor *t) { return t->size / cols_of(t); }

void *perm_alloc(i64 bytes); // zeroed
void perm_reset(void);       // frees every param, only tests need this
