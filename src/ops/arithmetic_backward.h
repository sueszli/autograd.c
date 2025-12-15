#pragma once

#include "autograd.h"
#include "tensor.h"

Tensor *tensor_add_backward_a(const Tensor *grad_output, const Tensor *a);
Tensor *tensor_add_backward_b(const Tensor *grad_output, const Tensor *b);
void add_backward(Function *fn, const Tensor *grad_output);

Tensor *tensor_sub_backward_a(const Tensor *grad_output, const Tensor *a);
Tensor *tensor_sub_backward_b(const Tensor *grad_output, const Tensor *b);
void sub_backward(Function *fn, const Tensor *grad_output);

Tensor *tensor_mul_backward_a(const Tensor *grad_output, const Tensor *a, const Tensor *b);
Tensor *tensor_mul_backward_b(const Tensor *grad_output, const Tensor *a, const Tensor *b);
void mul_backward(Function *fn, const Tensor *grad_output);

Tensor *tensor_div_backward_a(const Tensor *grad_output, const Tensor *a, const Tensor *b);
Tensor *tensor_div_backward_b(const Tensor *grad_output, const Tensor *a, const Tensor *b);
void div_backward(Function *fn, const Tensor *grad_output);

Tensor *tensor_matmul_backward_a(const Tensor *grad_output, const Tensor *a, const Tensor *b);
Tensor *tensor_matmul_backward_b(const Tensor *grad_output, const Tensor *a, const Tensor *b);
void matmul_backward(Function *fn, const Tensor *grad_output);
