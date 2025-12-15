#pragma once

#include "tensor.h"
#include <stdbool.h>

struct Tensor;

// gradient function node in the computational graph
typedef struct GradFn {
    void (*apply)(struct GradFn *self, const struct Tensor *grad_output); // backward pass function
    void (*destroy)(struct GradFn *self);                                 // destructor for subclass-specific cleanup
    struct GradFn **next_fns;                                             // gradient functions for inputs
    int64_t num_next;                                                     // number of inputs
    struct Tensor *out_tensor;                                            // output tensor
    char *name;                                                           // operation name for debugging
} GradFn;

// performs backpropagation from root tensor
void backward(struct Tensor *root, const struct Tensor *grad);

// initializes gradient function
void grad_fn_init(GradFn *fn, void (*apply)(GradFn *, const struct Tensor *), void (*destroy)(GradFn *), GradFn **next_fns, int64_t num_next, const char *name);

// frees gradient function resources
void grad_fn_free(GradFn *fn);

//
// backward constructors
//

GradFn *new_add_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_sub_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_mul_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_div_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_sum_backward(struct Tensor *input, int64_t dim_idx, bool keepdims);
GradFn *new_matmul_backward(struct Tensor *a, struct Tensor *b);
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
GradFn *new_tanh_backward(struct Tensor *input, struct Tensor *output);
GradFn *new_mean_backward(struct Tensor *input, int64_t dim_idx, bool keepdims);
GradFn *new_max_backward(struct Tensor *input, struct Tensor *output, int64_t dim_idx, bool keepdims);
GradFn *new_conv2d_backward(struct Tensor *input, struct Tensor *weight, struct Tensor *bias, uint64_t stride, uint64_t padding, uint64_t kernel_size);
GradFn *new_maxpool2d_backward(struct Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding);
GradFn *new_avgpool2d_backward(struct Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding);
GradFn *new_batchnorm2d_backward(struct Tensor *input, struct Tensor *gamma, struct Tensor *batch_mean, struct Tensor *batch_var, float32_t eps);
