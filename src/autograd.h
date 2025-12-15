#pragma once

#include "tensor.h"
#include <stdbool.h>

struct Tensor;

typedef struct GradFn {
    void (*apply)(struct GradFn *self, const struct Tensor *grad_output);
    struct GradFn **next_fns;
    int num_next;
    struct Tensor *out_tensor;
    char *name;
} GradFn;

void backward(struct Tensor *root, const struct Tensor *grad);

void grad_fn_init(GradFn *fn, void (*apply)(GradFn *, const struct Tensor *), GradFn **next_fns, int num_next, const char *name);
void grad_fn_free(GradFn *fn);

GradFn *new_add_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_sub_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_mul_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_div_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_matmul_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_sum_backward(struct Tensor *input, int64_t dim_idx, bool keepdims);
GradFn *new_relu_backward(struct Tensor *input);
GradFn *new_sigmoid_backward(struct Tensor *input, struct Tensor *output);
GradFn *new_softmax_backward(struct Tensor *input, struct Tensor *output, int64_t dim);
GradFn *new_reshape_backward(struct Tensor *input, const uint64_t *old_shape, uint64_t old_ndim);
GradFn *new_transpose_backward(struct Tensor *input, uint64_t dim0, uint64_t dim1);
GradFn *new_getitem_backward(struct Tensor *input, const uint64_t *multidim);
GradFn *new_gelu_backward(struct Tensor *input);
GradFn *new_mse_backward(struct Tensor *predictions, struct Tensor *targets);
GradFn *new_bce_backward(struct Tensor *predictions, struct Tensor *targets);
GradFn *new_crossentropy_backward(struct Tensor *logits, struct Tensor *targets);
