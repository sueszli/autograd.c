#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// 2-D row-major float64, the only shape microGPT needs
typedef struct Tensor {
    double *data;
    double *grad; // NULL until backward reaches it
    int64_t rows;
    int64_t cols;
    bool requires_grad;
} Tensor;

Tensor *tensor_new(int64_t rows, int64_t cols, const double *data, bool requires_grad); // data NULL gives zeros
void tensor_free(Tensor *t);
