#pragma once

#include "autograd.h"
#include "tensor.h"

Tensor *mse_loss_backward(const Tensor *predictions, const Tensor *targets);
void mse_loss_backward_fn(Function *fn, const Tensor *grad_output);

Tensor *cross_entropy_loss_backward(const Tensor *logits, const Tensor *targets);
void cross_entropy_loss_backward_fn(Function *fn, const Tensor *grad_output);

Tensor *binary_cross_entropy_loss_backward(const Tensor *predictions, const Tensor *targets);
void binary_cross_entropy_loss_backward_fn(Function *fn, const Tensor *grad_output);
