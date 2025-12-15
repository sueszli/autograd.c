#pragma once

#include "autograd.h"
#include "tensor.h"

Tensor *tensor_reshape_backward(const Tensor *grad_output, const Tensor *input);
typedef struct {
    uint64_t shape[MAX_NDIM];
    uint64_t ndim;
} ReshapeContext;
void reshape_backward(Function *fn, const Tensor *grad_output);

Tensor *tensor_transpose_backward(const Tensor *grad_output, uint64_t dim0, uint64_t dim1);
typedef struct {
    uint64_t dim0;
    uint64_t dim1;
} TransposeContext;
void transpose_backward(Function *fn, const Tensor *grad_output);
