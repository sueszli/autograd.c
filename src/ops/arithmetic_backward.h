#pragma once

#include "autograd.h"
#include "tensor.h"

Tensor *tensor_add_backward_a(const Tensor *grad_output, const Tensor *a);
Tensor *tensor_add_backward_b(const Tensor *grad_output, const Tensor *b);

Tensor *tensor_sub_backward_a(const Tensor *grad_output, const Tensor *a);
Tensor *tensor_sub_backward_b(const Tensor *grad_output, const Tensor *b);

Tensor *tensor_mul_backward_a(const Tensor *grad_output, const Tensor *a, const Tensor *b);
Tensor *tensor_mul_backward_b(const Tensor *grad_output, const Tensor *a, const Tensor *b);

Tensor *tensor_div_backward_a(const Tensor *grad_output, const Tensor *a, const Tensor *b);
Tensor *tensor_div_backward_b(const Tensor *grad_output, const Tensor *a, const Tensor *b);

Tensor *tensor_matmul_backward_a(const Tensor *grad_output, const Tensor *a, const Tensor *b);
Tensor *tensor_matmul_backward_b(const Tensor *grad_output, const Tensor *a, const Tensor *b);
