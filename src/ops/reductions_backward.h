#pragma once

#include "autograd.h"
#include "tensor.h"
#include <stdbool.h>

Tensor *tensor_sum_backward(const Tensor *grad_output, const Tensor *t, int64_t dim_idx, bool keepdims);
typedef struct {
    int64_t dim_idx;
    bool keepdims;
} SumContext;
void sum_backward(Function *fn, const Tensor *grad_output);

Tensor *tensor_mean_backward(const Tensor *grad_output, const Tensor *t, int64_t dim_idx, bool keepdims);
typedef struct {
    int64_t dim_idx;
    bool keepdims;
} MeanContext;
void mean_backward(Function *fn, const Tensor *grad_output);

Tensor *tensor_max_backward(const Tensor *grad_output, const Tensor *t, const Tensor *out, int64_t dim_idx, bool keepdims);
typedef struct {
    int64_t dim_idx;
    bool keepdims;
    Tensor *output;
} MaxContext;
void max_backward(Function *fn, const Tensor *grad_output);
