#pragma once

#include "tensor.h"

float32_t mse_loss(const Tensor *predictions, const Tensor *targets);
float32_t cross_entropy_loss(const Tensor *logits, const Tensor *targets);
float32_t binary_cross_entropy_loss(const Tensor *predictions, const Tensor *targets);
