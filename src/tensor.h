#pragma once

#include "autograd.h"
#include "utils/types.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define MAX_NDIM 32
#define MAX_TENSOR_SIZE (UINT64_MAX / sizeof(float32_t))

typedef struct Tensor {
    float32_t *data;   // flat contiguous array, row-major
    uint64_t *shape;   // array of dimension sizes
    uint64_t *strides; // array of elements to skip to get to next element in each dimension
    uint64_t ndim;     // rank (ie. 1 for vector, 2 for matrix, etc.)
    uint64_t size;     // total number of elements

    // autograd fields
    bool requires_grad;  // whether to track operations for autograd
    struct Tensor *grad; // accumulated gradient (del loss / del tensor) during backprop
    Function *grad_fn;   // function that created this tensor (NULL for leaves)
    uint32_t ref_count;  // reference count for memory management
} Tensor;

// internals
void linear_to_multidim_mut(uint64_t lin, const uint64_t *shape, uint64_t ndim, uint64_t *out_multidim);
uint64_t multidim_to_linear(const uint64_t *target, uint64_t target_ndim, const uint64_t *shape, uint64_t ndim, const uint64_t *strides);

// memory management
Tensor *tensor_create(const float32_t *data, const uint64_t *shape, uint64_t ndim, bool requires_grad);
Tensor *tensor_zeros(const uint64_t *shape, uint64_t ndim, bool requires_grad);
void tensor_free(Tensor *t);
Tensor *tensor_retain(Tensor *t);
void tensor_release(Tensor *t);
void tensor_zero_grad(Tensor *t);

// utils
void tensor_print(const Tensor *t);
Tensor *tensor_get(const Tensor *t, const uint64_t *multidim);
