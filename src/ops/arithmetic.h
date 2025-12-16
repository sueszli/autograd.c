#pragma once

#include "tensor.h"
#include <stdbool.h>

Tensor *tensor_add(const Tensor *a, const Tensor *b);
Tensor *tensor_sub(const Tensor *a, const Tensor *b);

// macro overloading to make the `disable_grad` arg optional (default to `false`)
#define tensor_mul(...) TENSOR_MUL_SELECT(__VA_ARGS__, tensor_mul_3, tensor_mul_2)(__VA_ARGS__)
#define TENSOR_MUL_SELECT(_1, _2, _3, NAME, ...) NAME
#define tensor_mul_2(a, b) tensor_mul_impl(a, b, false)
#define tensor_mul_3(a, b, disable_grad) tensor_mul_impl(a, b, disable_grad)
Tensor *tensor_mul_impl(const Tensor *a, const Tensor *b, bool disable_grad);

Tensor *tensor_div(const Tensor *a, const Tensor *b);
Tensor *tensor_matmul(const Tensor *a, const Tensor *b);
