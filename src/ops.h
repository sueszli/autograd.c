#pragma once

#include "tensor.h"

//
// the 12 ops microGPT needs, named after pytorch.
// https://pytorch.org/docs/stable/torch.html
//
// row-wise ops act on the last dim.
// matmul, transpose and the mask act on the last two dims, and leading dims are a batch.
// each file in ops/ keeps an op's forward and backward next to each other.
//

// ops/elementwise.c
Tensor *add(Tensor *a, Tensor *b); // a + b, same shape
Tensor *mul(Tensor *a, f64 s);     // a * s
Tensor *relu(Tensor *a);           // max(a, 0)

// ops/matrix.c
Tensor *matmul(Tensor *a, Tensor *b); // (..., n, k) @ (k, m) -> (..., n, m)
Tensor *transpose(Tensor *a);         // swaps the last two dims

// ops/indexing.c
Tensor *embedding(const i64 *ids, i64 n, Tensor *table); // rows of a 2-d table picked by ids, (n, d)
Tensor *masked_fill_upper(Tensor *a, f64 value);         // entries above the diagonal set to value, last two dims square
Tensor *narrow(Tensor *a, i64 start, i64 len);           // last dim [start, start + len)
Tensor *cat(Tensor *const *parts, i64 n);                // parts joined along the last dim

// ops/normalization.c
Tensor *rms_norm(Tensor *a, f64 eps);                                       // each row divided by its root mean square
Tensor *softmax(Tensor *a);                                                 // softmax over the last dim
Tensor *cross_entropy(Tensor *logits, const i64 *targets, const f64 *mask); // scalar, mean over rows with mask 1
