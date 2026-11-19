#include "linalg.h"

void linalg_matmul(f64 *out, const f64 *a, const f64 *b, i64 n, i64 k, i64 m) {
    // i-k-j order, so the inner loop runs along rows of b and out
    for (i64 i = 0; i < n; i++) {
        for (i64 p = 0; p < k; p++) {
            for (i64 j = 0; j < m; j++) {
                out[i * m + j] += a[i * k + p] * b[p * m + j];
            }
        }
    }
}

void linalg_matmul_transposed_a(f64 *out, const f64 *a, const f64 *b, i64 n, i64 k, i64 m) {
    // the shared row index i is outermost, so each out element still sums over i in order
    for (i64 i = 0; i < n; i++) {
        for (i64 p = 0; p < k; p++) {
            for (i64 j = 0; j < m; j++) {
                out[p * m + j] += a[i * k + p] * b[i * m + j];
            }
        }
    }
}

void linalg_transpose(f64 *out, const f64 *a, i64 n, i64 m) {
    // reads a row by row
    for (i64 i = 0; i < n; i++) {
        for (i64 j = 0; j < m; j++) {
            out[j * n + i] = a[i * m + j];
        }
    }
}
