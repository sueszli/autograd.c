#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef float float32_t;

typedef struct Tensor {
    float32_t* data;
    int64_t* shape;
    int64_t* strides;
    int64_t ndim;
    int64_t size;
    bool requires_grad;
    struct Tensor* grad;
} Tensor;

// creation and destruction
Tensor* tensor_create(float32_t* data, int64_t* shape, int64_t ndim, bool requires_grad);
Tensor* tensor_zeros(int64_t* shape, int64_t ndim, bool requires_grad);
void tensor_free(Tensor* tensor);

// utils
void tensor_print(Tensor* tensor);

// arithmetic
Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_sub(Tensor* a, Tensor* b);
Tensor* tensor_mul(Tensor* a, Tensor* b);
Tensor* tensor_div(Tensor* a, Tensor* b);

// matrix multiplication
Tensor* tensor_matmul(Tensor* a, Tensor* b);

// shape manipulation
Tensor* tensor_reshape(Tensor* tensor, int64_t* new_shape, int64_t new_ndim);
Tensor* tensor_transpose(Tensor* tensor, int64_t dim0, int64_t dim1);

// reductions
Tensor* tensor_sum(Tensor* tensor, int64_t axis, bool keepdims);
Tensor* tensor_mean(Tensor* tensor, int64_t axis, bool keepdims);
Tensor* tensor_max(Tensor* tensor, int64_t axis, bool keepdims);
Tensor* tensor_get(Tensor* tensor, int64_t* indices);
