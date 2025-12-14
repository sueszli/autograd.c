#pragma once

#include <stdbool.h>
#include <stddef.h>

typedef struct Tensor {
    float* data;
    int* shape;
    int* strides;
    int ndim;
    int size;
    bool requires_grad;
    struct Tensor* grad;
} Tensor;

// Creation and Destruction
Tensor* tensor_create(float* data, int* shape, int ndim, bool requires_grad);
Tensor* tensor_zeros(int* shape, int ndim, bool requires_grad);
void tensor_free(Tensor* tensor);

// Utils
void tensor_print(Tensor* tensor);

// Arithmetic
Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_sub(Tensor* a, Tensor* b);
Tensor* tensor_mul(Tensor* a, Tensor* b);
Tensor* tensor_div(Tensor* a, Tensor* b);

// Matrix Multiplication
Tensor* tensor_matmul(Tensor* a, Tensor* b);

// Shape Manipulation
Tensor* tensor_reshape(Tensor* tensor, int* new_shape, int new_ndim);
Tensor* tensor_transpose(Tensor* tensor, int dim0, int dim1);

// Reductions
Tensor* tensor_sum(Tensor* tensor, int axis, bool keepdims);
Tensor* tensor_mean(Tensor* tensor, int axis, bool keepdims);
Tensor* tensor_max(Tensor* tensor, int axis, bool keepdims);
Tensor* tensor_get(Tensor* tensor, int* indices); // Element access for testing/debugging
