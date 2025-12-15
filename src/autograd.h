#pragma once

#include "tensor.h"
#include <stdbool.h>

struct Tensor;

typedef struct GradFn {
    // computes gradients and accumulates them into input tensors
    void (*apply)(struct GradFn *self, const struct Tensor *grad_output);

    // references to the parents in the graph.
    // we do NOT own these pointers (they are owned by the Tensors or by the graph structure implied by Tensors)
    struct GradFn **next_fns;
    int num_next;
    struct Tensor *out_tensor;
    char *name;
} GradFn;

void backward(struct Tensor *root, const struct Tensor *grad);

void grad_fn_init(GradFn *fn, void (*apply)(GradFn *, const struct Tensor *), GradFn **next_fns, int num_next, const char *name);

void grad_fn_free(GradFn *fn);

// Backward function constructors
GradFn *new_add_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_sub_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_mul_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_div_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_matmul_backward(struct Tensor *a, struct Tensor *b);

GradFn *new_sum_backward(struct Tensor *input, int64_t dim_idx, bool keepdims);

GradFn *new_relu_backward(struct Tensor *input);
GradFn *new_sigmoid_backward(struct Tensor *input, struct Tensor *output);
GradFn *new_softmax_backward(struct Tensor *input, struct Tensor *output, int64_t dim);
