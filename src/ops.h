#pragma once

#include "tensor.h"

Tensor *add(Tensor *a, Tensor *b);                            // same shape
Tensor *matmul(Tensor *a, Tensor *b);                         // (n, k) @ (k, m)
Tensor *transpose(Tensor *a);                                 // (n, m) -> (m, n)
Tensor *gather(Tensor *table, const int64_t *ids, int64_t n); // rows of table picked by ids
