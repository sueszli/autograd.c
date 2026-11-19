#include "linalg.h"

void linalg_matmul(double *out, const double *a, const double *b, int64_t n, int64_t k, int64_t m) {
    // i-k-j order, so the inner loop runs along rows of b and out
    for (int64_t i = 0; i < n; i++) {
        for (int64_t p = 0; p < k; p++) {
            for (int64_t j = 0; j < m; j++) {
                out[i * m + j] += a[i * k + p] * b[p * m + j];
            }
        }
    }
}

void linalg_matmul_transposed_a(double *out, const double *a, const double *b, int64_t n, int64_t k, int64_t m) {
    // the shared row index i is outermost, so each out element still sums over i in order
    for (int64_t i = 0; i < n; i++) {
        for (int64_t p = 0; p < k; p++) {
            for (int64_t j = 0; j < m; j++) {
                out[p * m + j] += a[i * k + p] * b[i * m + j];
            }
        }
    }
}

void linalg_transpose(double *out, const double *a, int64_t n, int64_t m) {
    // reads a row by row
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < m; j++) {
            out[j * n + i] = a[i * m + j];
        }
    }
}
