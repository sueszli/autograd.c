#pragma once

#include "tensor.h"

//
// internals, the 12 ops behind the tensor namespace.
// each file in ops/ keeps an op's forward above its backward.
//

// ops/elementwise.c
[[nodiscard]] Tensor *op_add(Tensor *a, Tensor *b);
[[nodiscard]] Tensor *op_mul(Tensor *a, f64 s);
[[nodiscard]] Tensor *op_relu(Tensor *a);

// ops/matrix.c
[[nodiscard]] Tensor *op_matmul(Tensor *a, Tensor *b);
[[nodiscard]] Tensor *op_transpose(Tensor *a);

// ops/indexing.c
[[nodiscard]] Tensor *op_embedding(const i64 *ids, i64 n, Tensor *table);
[[nodiscard]] Tensor *op_masked_fill_upper(Tensor *a, f64 value);
[[nodiscard]] Tensor *op_narrow(Tensor *a, i64 start, i64 len);
[[nodiscard]] Tensor *op_cat(Tensor *const *parts, i64 n);

// ops/normalization.c
[[nodiscard]] Tensor *op_rms_norm(Tensor *a, f64 eps);
[[nodiscard]] Tensor *op_softmax(Tensor *a);
[[nodiscard]] Tensor *op_cross_entropy(Tensor *logits, const i64 *targets, const f64 *mask);
