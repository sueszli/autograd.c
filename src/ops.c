#include "ops.h"
#include "autograd.h"
#include <string.h>

//
// add
//

static void add_backward(const Node *node) {
    const Tensor *out = node->out;
    for (int64_t i = 0; i < node->n_in; i++) {
        double *g = grad_of(node->in[i]);
        for (int64_t j = 0; g != NULL && j < out->rows * out->cols; j++) {
            g[j] += out->grad[j];
        }
    }
}

Tensor *add(Tensor *a, Tensor *b) {
    CHECK(a->rows == b->rows && a->cols == b->cols, "add: shape mismatch");
    Tensor *out = tape_push(a->rows, a->cols, add_backward, 2, (Tensor *[]){a, b}, NULL);
    for (int64_t i = 0; i < a->rows * a->cols; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return out;
}

//
// matmul
//

static void matmul_backward(const Node *node) {
    const Tensor *a = node->in[0], *b = node->in[1], *out = node->out;
    int64_t n = a->rows, k = a->cols, m = b->cols;
    double *ga = grad_of(node->in[0]);
    double *gb = grad_of(node->in[1]);
    for (int64_t i = 0; ga != NULL && i < n; i++) {
        for (int64_t p = 0; p < k; p++) {
            for (int64_t j = 0; j < m; j++) {
                ga[i * k + p] += out->grad[i * m + j] * b->data[p * m + j];
            }
        }
    }
    for (int64_t i = 0; gb != NULL && i < n; i++) {
        for (int64_t p = 0; p < k; p++) {
            for (int64_t j = 0; j < m; j++) {
                gb[p * m + j] += a->data[i * k + p] * out->grad[i * m + j];
            }
        }
    }
}

// i-k-j order keeps the inner loop contiguous and sums each output over k in ascending order
Tensor *matmul(Tensor *a, Tensor *b) {
    CHECK(a->cols == b->rows, "matmul: inner dims mismatch");
    int64_t n = a->rows, k = a->cols, m = b->cols;
    Tensor *out = tape_push(n, m, matmul_backward, 2, (Tensor *[]){a, b}, NULL);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t p = 0; p < k; p++) {
            for (int64_t j = 0; j < m; j++) {
                out->data[i * m + j] += a->data[i * k + p] * b->data[p * m + j];
            }
        }
    }
    return out;
}

//
// transpose
//

static void transpose_backward(const Node *node) {
    const Tensor *out = node->out;
    double *g = grad_of(node->in[0]);
    for (int64_t i = 0; g != NULL && i < out->rows; i++) {
        for (int64_t j = 0; j < out->cols; j++) {
            g[j * out->rows + i] += out->grad[i * out->cols + j];
        }
    }
}

Tensor *transpose(Tensor *a) {
    Tensor *out = tape_push(a->cols, a->rows, transpose_backward, 1, (Tensor *[]){a}, NULL);
    for (int64_t i = 0; i < a->rows; i++) {
        for (int64_t j = 0; j < a->cols; j++) {
            out->data[j * a->rows + i] = a->data[i * a->cols + j];
        }
    }
    return out;
}

//
// gather
//

// scatter rows back by id, summing duplicates
static void gather_backward(const Node *node) {
    const Tensor *out = node->out;
    const int64_t *ids = node->ctx;
    double *g = grad_of(node->in[0]);
    for (int64_t r = 0; g != NULL && r < out->rows; r++) {
        for (int64_t c = 0; c < out->cols; c++) {
            g[ids[r] * out->cols + c] += out->grad[r * out->cols + c];
        }
    }
}

Tensor *gather(Tensor *table, const int64_t *ids, int64_t n) {
    CHECK(n > 0, "gather: no ids");
    int64_t *ids_copy = malloc((size_t)n * sizeof(int64_t));
    CHECK(ids_copy != NULL, "malloc failed");
    for (int64_t r = 0; r < n; r++) {
        CHECK(ids[r] >= 0 && ids[r] < table->rows, "gather: id out of range");
        ids_copy[r] = ids[r];
    }
    Tensor *out = tape_push(n, table->cols, gather_backward, 1, (Tensor *[]){table}, ids_copy);
    for (int64_t r = 0; r < n; r++) {
        memcpy(&out->data[r * table->cols], &table->data[ids[r] * table->cols], (size_t)table->cols * sizeof(double));
    }
    return out;
}
