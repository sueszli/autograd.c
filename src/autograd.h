#pragma once

#include "tensor.h"
#include <stdbool.h>

struct Tensor;

typedef struct GradFn {
    // computes gradients and accumulates them into input tensors
    void (*apply)(struct GradFn *self, struct Tensor *grad_output);

    // references to the parents in the graph.
    // we do NOT own these pointers (they are owned by the Tensors or by the graph structure implied by Tensors)
    struct GradFn **next_fns;
    int num_next;
    struct Tensor *out_tensor;
    char *name;
} GradFn;

void backward(struct Tensor *root, struct Tensor *grad);

void grad_fn_init(GradFn *fn, void (*apply)(GradFn *, struct Tensor *), GradFn **next_fns, int num_next, char *name);

void grad_fn_free(GradFn *fn);

// todo: to match current style, call them add_backward_create etc.
GradFn *new_add_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_sub_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_mul_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_div_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_matmul_backward(struct Tensor *a, struct Tensor *b);

GradFn *new_sum_backward(struct Tensor *input, int64_t dim_idx, bool keepdims);

GradFn *new_relu_backward(struct Tensor *input);
GradFn *new_sigmoid_backward(struct Tensor *input, struct Tensor *output);
GradFn *new_softmax_backward(struct Tensor *input, struct Tensor *output, int64_t dim);
