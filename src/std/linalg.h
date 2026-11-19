#pragma once

#include "types.h"

//
// dense row-major matrix kernels on plain f64 arrays.
// every inner loop walks memory contiguously.
// every output element sums its terms in ascending order, so results are reproducible.
// https://numpy.org/doc/stable/reference/routines.linalg.html
//

// out (n x m) += a (n x k) @ b (k x m)
void linalg_matmul(f64 *out, const f64 *a, const f64 *b, i64 n, i64 k, i64 m);

// out (k x m) += a^T @ b, for a (n x k) and b (n x m), without copying a^T
void linalg_matmul_transposed_a(f64 *out, const f64 *a, const f64 *b, i64 n, i64 k, i64 m);

// out (m x n) = a^T, for a (n x m)
void linalg_transpose(f64 *out, const f64 *a, i64 n, i64 m);
