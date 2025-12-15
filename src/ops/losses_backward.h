#pragma once

#include "tensor.h"

Tensor *mse_loss_backward(const Tensor *predictions, const Tensor *targets);
// todo: autograd backward

Tensor *cross_entropy_loss_backward(const Tensor *logits, const Tensor *targets);
// todo: autograd backward

Tensor *binary_cross_entropy_loss_backward(const Tensor *predictions, const Tensor *targets);
// todo: autograd backward
