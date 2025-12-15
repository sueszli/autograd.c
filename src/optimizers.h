#pragma once

#include "tensor.h"
#include <stdbool.h>

// optimizer base type using vtable polymorphism
typedef struct Optimizer Optimizer;
typedef void (*OptimizerStepFunc)(Optimizer *opt);
typedef void (*OptimizerFreeFunc)(Optimizer *opt);

struct Optimizer {
    Tensor **params;    // parameters being optimized
    size_t param_count; // number of parameters
    size_t step_count;  // optimization steps taken

    OptimizerStepFunc step; // update rule implementation
    OptimizerFreeFunc free; // cleanup implementation
};

void optimizer_zero_grad(Optimizer *opt); // zeros all parameter gradients
void optimizer_step(Optimizer *opt);      // updates parameters
void optimizer_free(Optimizer *opt);      // frees optimizer

//
// initializers
//

// sgd: v = momentum * v - lr * grad; param += v
Optimizer *optimizer_sgd_create(Tensor **params, size_t count, float32_t lr, float32_t momentum, float32_t weight_decay);

// adam: adaptive moment estimation
Optimizer *optimizer_adam_create(Tensor **params, size_t count, float32_t lr, float32_t beta1, float32_t beta2, float32_t eps, float32_t weight_decay);

// adamw: adam with decoupled weight decay
Optimizer *optimizer_adamw_create(Tensor **params, size_t count, float32_t lr, float32_t beta1, float32_t beta2, float32_t eps, float32_t weight_decay);
