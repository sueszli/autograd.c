#pragma once

#include "tensor.h"

// adam without weight decay and with linear learning rate decay to 0, as in microGPT
typedef struct {
    Tensor **params;
    i64 n;
    f64 **m, **v;
} AdamW;

AdamW adamw_new(Tensor **params, i64 n);              // m and v in perm
void adamw_step(AdamW *opt, i64 step, i64 num_steps); // step counts from 1, zeroes grads after the update
