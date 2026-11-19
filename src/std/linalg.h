#pragma once

#include <stdint.h>

//
// dense row-major matrix kernels on plain double arrays.
// every inner loop walks memory contiguously.
// every output element sums its terms in ascending order, so results are reproducible.
// https://numpy.org/doc/stable/reference/routines.linalg.html
//

// out (n x m) += a (n x k) @ b (k x m)
void linalg_matmul(double *out, const double *a, const double *b, int64_t n, int64_t k, int64_t m);

// out (k x m) += a^T @ b, for a (n x k) and b (n x m), without copying a^T
void linalg_matmul_transposed_a(double *out, const double *a, const double *b, int64_t n, int64_t k, int64_t m);

// out (m x n) = a^T, for a (n x m)
void linalg_transpose(double *out, const double *a, int64_t n, int64_t m);
