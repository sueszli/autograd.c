#pragma once

#include <stdbool.h>
#include <stddef.h>

typedef struct Tensor {
    float *data;
    int *shape;
    int *strides;
    int ndim;
    int size;
    bool requires_grad;
    struct Tensor *grad;
} Tensor;

// Creation and Destruction
Tensor *tensor_create(const float *data, const int *shape, int ndim, bool requires_grad);
Tensor *tensor_zeros(const int *shape, int ndim, bool requires_grad);
void tensor_free(Tensor *t);

// Utils
void tensor_print(Tensor *t);

// Arithmetic
Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_sub(Tensor *a, Tensor *b);
Tensor *tensor_mul(Tensor *a, Tensor *b);
Tensor *tensor_div(Tensor *a, Tensor *b);

// Matrix Multiplication
Tensor *tensor_matmul(Tensor *a, Tensor *b);

// Shape Manipulation
Tensor *tensor_reshape(const Tensor *t, const int *new_shape, int new_ndim);
Tensor *tensor_transpose(Tensor *t, int dim0, int dim1);

// Reductions
Tensor *tensor_sum(Tensor *t, int axis, bool keepdims);
Tensor *tensor_mean(Tensor *t, int axis, bool keepdims);
Tensor *tensor_max(Tensor *t, int axis, bool keepdims);
Tensor *tensor_get(Tensor *t, const int *indices); // Element access for testing/debugging