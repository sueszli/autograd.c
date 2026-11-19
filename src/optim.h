#pragma once

#include "tensor.h"

// adam without weight decay and with linear learning rate decay to 0, as in microGPT
typedef struct {
    Tensor **params;
    i64 n;
    f64 **m, **v;
} AdamW;

typedef struct {
    AdamW (*new)(Tensor **params, i64 n);              // m and v in the perm arena of params[0]
    void (*step)(AdamW *opt, i64 step, i64 num_steps); // step counts from 1, zeroes grads after the update
} AdamwNamespace;

extern const AdamwNamespace adamw;
