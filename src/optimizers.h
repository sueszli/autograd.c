#pragma once

#include "tensor.h"
#include <stdbool.h>

// vtable polymorphism
typedef struct Optimizer Optimizer;
typedef void (*OptimizerStepFunc)(Optimizer *opt);
typedef void (*OptimizerFreeFunc)(Optimizer *opt);

struct Optimizer {
    Tensor **params;    // array of pointers to tensors being optimized
    size_t param_count; // number of parameters
    size_t step_count;  // number of optimization steps taken

    OptimizerStepFunc step; // implementation of the update rule
    OptimizerFreeFunc free; // implementation of resource cleanup
};

// resets gradients of all parameters to zero (frees the grad tensors).
// should be called before backward pass.
void optimizer_zero_grad(Optimizer *opt);

// performs a single optimization step (updates parameters).
void optimizer_step(Optimizer *opt);

// frees the optimizer and its internal resources.
// note: does NOT free the parameters themselves, only internal buffers.
void optimizer_free(Optimizer *opt);

//
// initializers
//

/**
 * creates an SGD optimizer.
 * updates:
 *   v = momentum * v - lr * grad
 *   param += v
 *
 * @param params       array of pointers to parameters to optimize
 * @param count        number of parameters
 * @param lr           learning rate
 * @param momentum     momentum factor (0.0 to disable)
 * @param weight_decay weight decay (L2 penalty) (0.0 to disable)
 * @return             pointer to new optimizer
 */
Optimizer *optimizer_sgd_create(Tensor **params, size_t count, float32_t lr, float32_t momentum, float32_t weight_decay);

/**
 * creates an Adam optimizer.
 *
 * @param params       array of pointers to parameters to optimize
 * @param count        number of parameters
 * @param lr           learning rate (default 0.001)
 * @param beta1        exponential decay rate for first moment estimates (default 0.9)
 * @param beta2        exponential decay rate for second moment estimates (default 0.999)
 * @param eps          term added to denominator to improve numerical stability (default 1e-8)
 * @param weight_decay weight decay (L2 penalty) (default 0.0)
 * @return             pointer to new optimizer
 */
Optimizer *optimizer_adam_create(Tensor **params, size_t count, float32_t lr, float32_t beta1, float32_t beta2, float32_t eps, float32_t weight_decay);

/**
 * creates an AdamW optimizer.
 * decoupling weight decay from the gradient update.
 *
 * @param params       array of pointers to parameters to optimize
 * @param count        number of parameters
 * @param lr           learning rate (default 0.001)
 * @param beta1        exponential decay rate for first moment estimates (default 0.9)
 * @param beta2        exponential decay rate for second moment estimates (default 0.999)
 * @param eps          term added to denominator to improve numerical stability (default 1e-8)
 * @param weight_decay weight decay (L2 penalty) (default 0.01)
 * @return             pointer to new optimizer
 */
Optimizer *optimizer_adamw_create(Tensor **params, size_t count, float32_t lr, float32_t beta1, float32_t beta2, float32_t eps, float32_t weight_decay);
