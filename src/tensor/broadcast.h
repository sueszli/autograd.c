/*
 * broadcasting means expanding the smaller tensor to match the larger one.
 * we follow the numpy/pytorch broadcasting convention.
 *
 * example: adding a 2x1 tensor to a 1x3 tensor results in a 2x3 tensor
 *
 *    A (2x1)       B (1x3)      Result (2x3)
 *    ┌─────┐    ┌──────────┐    ┌──────────┐
 *    │  1  │ +  │ 10 20 30 │ =  │ 11 21 31 │
 *    │  4  │    └──────────┘    │ 14 24 34 │
 *    └─────┘                    └──────────┘
 *
 * steps:
 *                       ┌─────┐   ┌───────────┐
 *    (1) A is expanded: │  1  │ → │ 1   1   1 │ (repeat across columns)
 *                       │  4  │   │ 4   4   4 │
 *                       └─────┘   └───────────┘
 *
 *                       ┌──────────┐   ┌──────────┐
 *    (2) B is expanded: │ 10 20 30 │ → │ 10 20 30 │ (repeat across rows)
 *                       └──────────┘   │ 10 20 30 │
 *                                      └──────────┘
 *
 *    (3) operation is performed element-wise
 */

#pragma once

#include "../utils/types.h"
#include "tensor.h"
#include <stdbool.h>

bool tensor_can_broadcast(const Tensor *a, const Tensor *b);

i32 *get_tensor_broadcast_shape(const Tensor *a, const Tensor *b, i32 *result_ndim);

Tensor *tensor_broadcast_to(const Tensor *tensor, const i32 *target_shape, i32 target_ndim);
