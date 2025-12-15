#pragma once

#include "tensor.h"

Tensor *mse_loss(const Tensor *predictions, const Tensor *targets);
Tensor *cross_entropy_loss(const Tensor *logits, const Tensor *targets);
Tensor *binary_cross_entropy_loss(const Tensor *predictions, const Tensor *targets);
