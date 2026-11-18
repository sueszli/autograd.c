#pragma once

#include "tensor.h"

Tensor *add(Tensor *a, Tensor *b);                                                        // same shape
Tensor *matmul(Tensor *a, Tensor *b);                                                     // (n, k) @ (k, m)
Tensor *transpose(Tensor *a);                                                             // (n, m) -> (m, n)
Tensor *gather(Tensor *table, const int64_t *ids, int64_t n);                             // rows of table picked by ids
Tensor *scale(Tensor *a, double s);                                                       // a * s
Tensor *relu(Tensor *a);                                                                  // max(a, 0)
Tensor *rmsnorm(Tensor *a, double eps);                                                   // each row divided by its root mean square
Tensor *causal_mask(Tensor *a, double value);                                             // square, entries above the diagonal set to value
Tensor *softmax_rows(Tensor *a);                                                          // softmax over each row
Tensor *slice_cols(Tensor *a, int64_t start, int64_t len);                                // columns [start, start + len)
Tensor *concat_cols(Tensor *const *parts, int64_t n);                                     // parts side by side, same rows
Tensor *masked_cross_entropy(Tensor *logits, const int64_t *targets, const double *mask); // 1x1 mean loss over rows with mask 1
