#include "ops.h"
#include "autograd.h"
#include <assert.h>
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
    assert(a->rows == b->rows && a->cols == b->cols && "add: shape mismatch");
    Tensor *out = node_new(a->rows, a->cols, add_backward, 2, (Tensor *[]){a, b}, 0, NULL);
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
    // ga += dout @ b^T, with b^T copied so the inner loop is contiguous and each output still sums over j in order
    double *bt = graph_alloc(k * m * (int64_t)sizeof(double));
    for (int64_t p = 0; ga != NULL && p < k; p++) {
        for (int64_t j = 0; j < m; j++) {
            bt[j * k + p] = b->data[p * m + j];
        }
    }
    for (int64_t i = 0; ga != NULL && i < n; i++) {
        for (int64_t j = 0; j < m; j++) {
            for (int64_t p = 0; p < k; p++) {
                ga[i * k + p] += out->grad[i * m + j] * bt[j * k + p];
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
    assert(a->cols == b->rows && "matmul: inner dims mismatch");
    int64_t n = a->rows, k = a->cols, m = b->cols;
    Tensor *out = node_new(n, m, matmul_backward, 2, (Tensor *[]){a, b}, 0, NULL);
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
    Tensor *out = node_new(a->cols, a->rows, transpose_backward, 1, (Tensor *[]){a}, 0, NULL);
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
    assert(n > 0 && "gather: no ids");
    int64_t *ids_copy = graph_alloc(n * (int64_t)sizeof(int64_t));
    for (int64_t r = 0; r < n; r++) {
        assert(ids[r] >= 0 && ids[r] < table->rows && "gather: id out of range");
        ids_copy[r] = ids[r];
    }
    Tensor *out = node_new(n, table->cols, gather_backward, 1, (Tensor *[]){table}, 0, ids_copy);
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
    Tensor *out = node_new(a->rows, a->cols, scale_backward, 1, (Tensor *[]){a}, s, NULL);
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
    Tensor *out = node_new(a->rows, a->cols, relu_backward, 1, (Tensor *[]){a}, 0, NULL);
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
    Tensor *out = node_new(a->rows, a->cols, rmsnorm_backward, 1, (Tensor *[]){a}, eps, NULL);
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
    assert(a->rows == a->cols && "causal_mask: must be square");
    Tensor *out = node_new(a->rows, a->cols, causal_mask_backward, 1, (Tensor *[]){a}, 0, NULL);
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

static void softmax_row(const double *x, double *y, int64_t cols) {
    double max = x[0];
    for (int64_t j = 1; j < cols; j++) {
        max = x[j] > max ? x[j] : max;
    }
    double sum = 0.0;
    for (int64_t j = 0; j < cols; j++) {
        y[j] = exp(x[j] - max);
        sum += y[j];
    }
    double inv = 1.0 / sum;
    for (int64_t j = 0; j < cols; j++) {
        y[j] *= inv;
    }
}

Tensor *softmax_rows(Tensor *a) {
    Tensor *out = node_new(a->rows, a->cols, softmax_rows_backward, 1, (Tensor *[]){a}, 0, NULL);
    for (int64_t i = 0; i < a->rows; i++) {
        softmax_row(&a->data[i * a->cols], &out->data[i * a->cols], a->cols);
    }
    return out;
}

//
// slice and concat columns, used to split and merge attention heads
//

static void slice_cols_backward(const Node *node) {
    const Tensor *a = node->in[0], *out = node->out;
    int64_t start = (int64_t)node->aux;
    double *g = grad_of(node->in[0]);
    for (int64_t i = 0; g != NULL && i < out->rows; i++) {
        for (int64_t j = 0; j < out->cols; j++) {
            g[i * a->cols + start + j] += out->grad[i * out->cols + j];
        }
    }
}

Tensor *slice_cols(Tensor *a, int64_t start, int64_t len) {
    assert(start >= 0 && len > 0 && start + len <= a->cols && "slice_cols: out of range");
    Tensor *out = node_new(a->rows, len, slice_cols_backward, 1, (Tensor *[]){a}, (double)start, NULL);
    for (int64_t i = 0; i < a->rows; i++) {
        memcpy(&out->data[i * len], &a->data[i * a->cols + start], (size_t)len * sizeof(double));
    }
    return out;
}

static void concat_cols_backward(const Node *node) {
    const Tensor *out = node->out;
    for (int64_t p = 0, start = 0; p < node->n_in; start += node->in[p]->cols, p++) {
        double *g = grad_of(node->in[p]);
        int64_t len = node->in[p]->cols;
        for (int64_t i = 0; g != NULL && i < out->rows; i++) {
            for (int64_t j = 0; j < len; j++) {
                g[i * len + j] += out->grad[i * out->cols + start + j];
            }
        }
    }
}

Tensor *concat_cols(Tensor *const *parts, int64_t n) {
    assert(n > 0 && "concat_cols: no parts");
    int64_t cols = 0;
    for (int64_t p = 0; p < n; p++) {
        assert(parts[p]->rows == parts[0]->rows && "concat_cols: row mismatch");
        cols += parts[p]->cols;
    }
    Tensor *out = node_new(parts[0]->rows, cols, concat_cols_backward, n, parts, 0, NULL);
    for (int64_t p = 0, start = 0; p < n; start += parts[p]->cols, p++) {
        for (int64_t i = 0; i < out->rows; i++) {
            memcpy(&out->data[i * cols + start], &parts[p]->data[i * parts[p]->cols], (size_t)parts[p]->cols * sizeof(double));
        }
    }
    return out;
}

//
// masked cross-entropy
//

/*
 * fused with its softmax: loss = (1 / sum(mask)) * sum_i -mask[i] * log(p[i, target[i]])
 * ctx holds targets followed by mask in one arena buffer.
 */
static void masked_cross_entropy_backward(const Node *node) {
    const Tensor *logits = node->in[0];
    int64_t rows = logits->rows, cols = logits->cols;
    const int64_t *targets = node->ctx;
    const double *mask = (const double *)(targets + rows);
    double *g = grad_of(node->in[0]);
    double sum_mask = 0.0;
    for (int64_t i = 0; i < rows; i++) {
        sum_mask += mask[i];
    }
    double inv = sum_mask == 0.0 ? 0.0 : 1.0 / sum_mask;
    double *p = graph_alloc(cols * (int64_t)sizeof(double));
    for (int64_t i = 0; g != NULL && i < rows; i++) {
        softmax_row(&logits->data[i * cols], p, cols);
        for (int64_t j = 0; j < cols; j++) {
            double onehot = j == targets[i] ? 1.0 : 0.0;
            g[i * cols + j] += node->out->grad[0] * (mask[i] * (p[j] - onehot) * inv);
        }
    }
}

Tensor *masked_cross_entropy(Tensor *logits, const int64_t *targets, const double *mask) {
    int64_t rows = logits->rows, cols = logits->cols;
    int64_t *ctx = graph_alloc(rows * (int64_t)(sizeof(int64_t) + sizeof(double)));
    double *mask_copy = (double *)(ctx + rows);
    for (int64_t i = 0; i < rows; i++) {
        assert(targets[i] >= 0 && targets[i] < cols && "masked_cross_entropy: target out of range");
        ctx[i] = targets[i];
        mask_copy[i] = mask[i];
    }
    Tensor *out = node_new(1, 1, masked_cross_entropy_backward, 1, (Tensor *[]){logits}, 0, ctx);
    double *p = graph_alloc(cols * (int64_t)sizeof(double));
    double total = 0.0, sum_mask = 0.0;
    for (int64_t i = 0; i < rows; i++) {
        softmax_row(&logits->data[i * cols], p, cols);
        total -= mask[i] * log(fmax(p[targets[i]], 1e-30));
        sum_mask += mask[i];
    }
    out->data[0] = sum_mask == 0.0 ? 0.0 : (1.0 / sum_mask) * total;
    return out;
}
