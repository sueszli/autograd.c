#pragma once

#include "autograd.h"
#include "tensor.h"

Tensor *tensor_sigmoid_backward(const Tensor *t);
void sigmoid_backward(Function *fn, const Tensor *grad_output);

Tensor *tensor_relu_backward(const Tensor *t);
void relu_backward(Function *fn, const Tensor *grad_output);

void tanh_backward(Function *fn, const Tensor *grad_output);
Tensor *tensor_tanh_backward(const Tensor *t);

Tensor *tensor_gelu_backward(const Tensor *t);
void gelu_backward(Function *fn, const Tensor *grad_output);

Tensor *tensor_softmax_backward(const Tensor *t, int64_t dim);
void softmax_backward(Function *fn, const Tensor *grad_output);
