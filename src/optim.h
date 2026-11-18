#pragma once

#include "tensor.h"

// adam without weight decay and with linear learning rate decay to 0, as in microGPT
typedef struct {
    Tensor **params;
    int64_t n;
    double **m, **v;
} AdamW;

AdamW adamw_new(Tensor **params, int64_t n);
void adamw_step(AdamW *opt, int64_t step, int64_t num_steps); // step counts from 1, zeroes grads after the update
void adamw_free(AdamW *opt);
