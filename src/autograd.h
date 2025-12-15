#pragma once

#include "tensor.h"
#include <stdbool.h>
#include "tensor_autograd.h"

struct Tensor;

// gradient function node in the computational graph
typedef struct GradFn {
    void (*apply)(struct GradFn *self, const struct Tensor *grad_output); // backward pass function
    void (*destroy)(struct GradFn *self);                                 // destructor for subclass-specific cleanup
    struct GradFn **next_fns;                                             // gradient functions for inputs
    int64_t next_fn_count;                                                // number of inputs
    struct Tensor *out_tensor;                                            // output tensor
    char *name;                                                           // operation name for debugging
} GradFn;

// performs backpropagation from root tensor
void backward(struct Tensor *root, const struct Tensor *grad);

// initializes gradient function
void grad_fn_init(GradFn *fn, void (*apply)(GradFn *, const struct Tensor *), void (*destroy)(GradFn *), GradFn **next_fns, int64_t next_fn_count, const char *name);

// frees gradient function resources
void grad_fn_free(GradFn *fn);

// accumulates gradient into tensor
void accumulate_grad(struct Tensor *tensor_mut, const struct Tensor *grad);
