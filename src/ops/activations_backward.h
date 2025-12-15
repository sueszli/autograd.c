#pragma once

#include "autograd.h"
#include "tensor.h"

Tensor *tensor_sigmoid_backward(const Tensor *t);
Tensor *tensor_relu_backward(const Tensor *t);
Tensor *tensor_tanh_backward(const Tensor *t);
Tensor *tensor_gelu_backward(const Tensor *t);
Tensor *tensor_softmax_backward(const Tensor *t, int64_t dim);

// Context structure for softmax
// typedef struct {
//     int64_t dim;
// } SoftmaxContext;

// // High-level backward callbacks (used by autograd system)
// void sigmoid_backward(Function *fn, const Tensor *grad_output);
// void relu_backward(Function *fn, const Tensor *grad_output);
// void tanh_backward(Function *fn, const Tensor *grad_output);
// void gelu_backward(Function *fn, const Tensor *grad_output);
// void softmax_backward(Function *fn, const Tensor *grad_output);
