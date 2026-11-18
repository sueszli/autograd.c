#include "ops.h"
#include "autograd.h"
#include <math.h>
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
    Tensor *out = tape_push(a->rows, a->cols, add_backward, 2, (Tensor *[]){a, b}, 0, NULL);
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
    Tensor *out = tape_push(n, m, matmul_backward, 2, (Tensor *[]){a, b}, 0, NULL);
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
    Tensor *out = tape_push(a->cols, a->rows, transpose_backward, 1, (Tensor *[]){a}, 0, NULL);
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
    Tensor *out = tape_push(n, table->cols, gather_backward, 1, (Tensor *[]){table}, 0, ids_copy);
    for (int64_t r = 0; r < n; r++) {
        memcpy(&out->data[r * table->cols], &table->data[ids[r] * table->cols], (size_t)table->cols * sizeof(double));
    }
    return out;
}

//
// scale
//

static void scale_backward(const Node *node) {
    const Tensor *out = node->out;
    double *g = grad_of(node->in[0]);
    for (int64_t i = 0; g != NULL && i < out->rows * out->cols; i++) {
        g[i] += node->aux * out->grad[i];
    }
}

Tensor *scale(Tensor *a, double s) {
    Tensor *out = tape_push(a->rows, a->cols, scale_backward, 1, (Tensor *[]){a}, s, NULL);
    for (int64_t i = 0; i < a->rows * a->cols; i++) {
        out->data[i] = a->data[i] * s;
    }
    return out;
}

//
// relu
//

static void relu_backward(const Node *node) {
    const Tensor *a = node->in[0], *out = node->out;
    double *g = grad_of(node->in[0]);
    for (int64_t i = 0; g != NULL && i < out->rows * out->cols; i++) {
        g[i] += a->data[i] > 0.0 ? out->grad[i] : 0.0;
    }
}

Tensor *relu(Tensor *a) {
    Tensor *out = tape_push(a->rows, a->cols, relu_backward, 1, (Tensor *[]){a}, 0, NULL);
    for (int64_t i = 0; i < a->rows * a->cols; i++) {
        out->data[i] = a->data[i] > 0.0 ? a->data[i] : 0.0;
    }
    return out;
}

//
// rmsnorm
//

// recomputed in backward instead of cached, same inputs give the same bits
static double rms_scale(const double *row, int64_t cols, double eps) {
    double sum_sq = 0.0;
    for (int64_t j = 0; j < cols; j++) {
        sum_sq += row[j] * row[j];
    }
    return pow(sum_sq * (1.0 / (double)cols) + eps, -0.5);
}

// dx = s * (dy - x * dot(dy, x) * s^2 / d)
static void rmsnorm_backward(const Node *node) {
    const Tensor *a = node->in[0], *out = node->out;
    double *g = grad_of(node->in[0]);
    for (int64_t i = 0; g != NULL && i < a->rows; i++) {
        const double *x = &a->data[i * a->cols], *dy = &out->grad[i * a->cols];
        double s = rms_scale(x, a->cols, node->aux);
        double dot = 0.0;
        for (int64_t j = 0; j < a->cols; j++) {
            dot += dy[j] * x[j];
        }
        for (int64_t j = 0; j < a->cols; j++) {
            g[i * a->cols + j] += s * (dy[j] - x[j] * dot * s * s / (double)a->cols);
        }
    }
}

// normalizes each row to unit root mean square, no learned weight
Tensor *rmsnorm(Tensor *a, double eps) {
    Tensor *out = tape_push(a->rows, a->cols, rmsnorm_backward, 1, (Tensor *[]){a}, eps, NULL);
    for (int64_t i = 0; i < a->rows; i++) {
        double s = rms_scale(&a->data[i * a->cols], a->cols, eps);
        for (int64_t j = 0; j < a->cols; j++) {
            out->data[i * a->cols + j] = a->data[i * a->cols + j] * s;
        }
    }
    return out;
}

//
// causal mask
//

static void causal_mask_backward(const Node *node) {
    const Tensor *out = node->out;
    double *g = grad_of(node->in[0]);
    for (int64_t i = 0; g != NULL && i < out->rows; i++) {
        for (int64_t j = 0; j <= i; j++) {
            g[i * out->cols + j] += out->grad[i * out->cols + j];
        }
    }
}

// row i may only attend to columns j <= i, the rest are set to value
Tensor *causal_mask(Tensor *a, double value) {
    CHECK(a->rows == a->cols, "causal_mask: must be square");
    Tensor *out = tape_push(a->rows, a->cols, causal_mask_backward, 1, (Tensor *[]){a}, 0, NULL);
    for (int64_t i = 0; i < a->rows; i++) {
        for (int64_t j = 0; j < a->cols; j++) {
            out->data[i * a->cols + j] = j <= i ? a->data[i * a->cols + j] : value;
        }
    }
    return out;
}

//
// softmax
//

// dx = y * (dy - dot(y, dy)), per row
static void softmax_rows_backward(const Node *node) {
    const Tensor *out = node->out;
    double *g = grad_of(node->in[0]);
    for (int64_t i = 0; g != NULL && i < out->rows; i++) {
        const double *y = &out->data[i * out->cols], *dy = &out->grad[i * out->cols];
        double dot = 0.0;
        for (int64_t j = 0; j < out->cols; j++) {
            dot += y[j] * dy[j];
        }
        for (int64_t j = 0; j < out->cols; j++) {
            g[i * out->cols + j] += y[j] * (dy[j] - dot);
        }
    }
}

// multiplies by 1/sum instead of dividing, to match the reference rounding
Tensor *softmax_rows(Tensor *a) {
    Tensor *out = tape_push(a->rows, a->cols, softmax_rows_backward, 1, (Tensor *[]){a}, 0, NULL);
    for (int64_t i = 0; i < a->rows; i++) {
        const double *x = &a->data[i * a->cols];
        double *y = &out->data[i * a->cols];
        double max = x[0];
        for (int64_t j = 1; j < a->cols; j++) {
            max = x[j] > max ? x[j] : max;
        }
        double sum = 0.0;
        for (int64_t j = 0; j < a->cols; j++) {
            y[j] = exp(x[j] - max);
            sum += y[j];
        }
        double inv = 1.0 / sum;
        for (int64_t j = 0; j < a->cols; j++) {
            y[j] *= inv;
        }
    }
    return out;
}
