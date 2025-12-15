#pragma once

#include "tensor.h"
#include <stdint.h>

// randomly flip images horizontally with given probability p
void random_horizontal_flip_mut(Tensor *t, float32_t p);

// randomly crop image to (target_h, target_w) after padding
void random_crop_mut(Tensor *t, uint64_t target_h, uint64_t target_w, uint64_t padding);
