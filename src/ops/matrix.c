#include "autograd.h"
#include "ops.h"
#include <assert.h>
#include <string.h>

static void matmul_backward(const Node *node);
static void transpose_backward(const Node *node);

// the kernels sum every output element in ascending order, so results are reproducible

static void kernel_matmul(f64 *out, const f64 *a, const f64 *b, i64 n, i64 k, i64 m) {
    // out (n x m) += a (n x k) @ b (k x m)
    // i-k-j order, so the inner loop runs along rows of b and out
    for (i64 i = 0; i < n; i++) {
        for (i64 p = 0; p < k; p++) {
            for (i64 j = 0; j < m; j++) {
                out[i * m + j] += a[i * k + p] * b[p * m + j];
            }
        }
    }
}

static void kernel_matmul_transposed_a(f64 *out, const f64 *a, const f64 *b, i64 n, i64 k, i64 m) {
    // out (k x m) += a^T @ b, for a (n x k) and b (n x m), without copying a^T
    // the shared row index i is outermost, so each out element still sums over i in order
    for (i64 i = 0; i < n; i++) {
        for (i64 p = 0; p < k; p++) {
            for (i64 j = 0; j < m; j++) {
                out[p * m + j] += a[i * k + p] * b[i * m + j];
            }
        }
    }
}

static void kernel_transpose(f64 *out, const f64 *a, i64 n, i64 m) {
    // out (m x n) = a^T, for a (n x m)
    for (i64 i = 0; i < n; i++) {
        for (i64 j = 0; j < m; j++) {
            out[j * n + i] = a[i * m + j];
        }
    }
}

Tensor *matmul(Tensor *a, Tensor *b) {
    // leading dims of a fold into rows, so a batch is one big matmul
    assert(a->ndim >= 1 && b->ndim == 2 && cols_of(a) == b->shape[0] && "matmul: shape mismatch");
    i64 shape[MAX_NDIM];
    memcpy(shape, a->shape, sizeof(shape));
    shape[a->ndim - 1] = b->shape[1];
    Tensor *out = node_new(shape, a->ndim, matmul_backward, 2, (Tensor *[]){a, b}, 0, NULL);
    kernel_matmul(out->data, a->data, b->data, rows_of(a), cols_of(a), b->shape[1]);
    return out;
}

static void matmul_backward(const Node *node) {
    // ga += dout @ b^T, with b^T copied so the inner loop stays contiguous
    // gb += a^T @ dout
    const Tensor *a = node->in[0], *b = node->in[1], *out = node->out;
    i64 n = rows_of(a), k = cols_of(a), m = cols_of(b);
    f64 *ga = grad_of(node->in[0]);
    f64 *gb = grad_of(node->in[1]);
    if (ga != NULL) {
        f64 *bt = graph_alloc(k * m * (i64)sizeof(f64));
        kernel_transpose(bt, b->data, k, m);
        kernel_matmul(ga, out->grad, bt, n, m, k);
    }
    if (gb != NULL) {
        kernel_matmul_transposed_a(gb, a->data, out->grad, n, k, m);
    }
}

Tensor *transpose(Tensor *a) {
    // copies, there are no strided views
    assert(a->ndim >= 2 && "transpose: needs at least 2 dims");
    i64 r = a->shape[a->ndim - 2], c = cols_of(a);
    i64 shape[MAX_NDIM];
    memcpy(shape, a->shape, sizeof(shape));
    shape[a->ndim - 2] = c;
    shape[a->ndim - 1] = r;
    Tensor *out = node_new(shape, a->ndim, transpose_backward, 1, (Tensor *[]){a}, 0, NULL);
    for (i64 off = 0; off < a->size; off += r * c) {
        kernel_transpose(&out->data[off], &a->data[off], r, c);
    }
    return out;
}

static void transpose_backward(const Node *node) {
    // each batch matrix of the grad, transposed back
    const Tensor *out = node->out;
    i64 r = out->shape[out->ndim - 2], c = cols_of(out);
    f64 *g = grad_of(node->in[0]);
    for (i64 off = 0; g != NULL && off < out->size; off += r * c) {
        for (i64 i = 0; i < r; i++) {
            for (i64 j = 0; j < c; j++) {
                g[off + j * r + i] += out->grad[off + i * c + j];
            }
        }
    }
}
