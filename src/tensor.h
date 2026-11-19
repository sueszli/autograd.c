#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// 2-D row-major float64, the only shape microGPT needs
typedef struct Tensor {
    double *data;
    double *grad;      // params get one in tensor_new, op outputs when backward reaches them
    struct Node *node; // the op that made this tensor, NULL for params and inputs
    int64_t rows;
    int64_t cols;
    bool requires_grad;
} Tensor;

Tensor *tensor_new(int64_t rows, int64_t cols, const double *data, bool requires_grad); // in perm_mem, data NULL gives zeros
