#include "autograd.h"
#include "ops.h"
#include <assert.h>
#include <math.h>

static f64 rms_scale(const f64 *row, i64 cols, f64 eps) {
    // 1 / sqrt(mean(x^2) + eps), recomputed in backward so the same inputs give the same bits
    f64 sum_sq = 0.0;
    for (i64 j = 0; j < cols; j++) {
        sum_sq += row[j] * row[j];
    }
    return pow(sum_sq * (1.0 / (f64)cols) + eps, -0.5);
}

static void rms_norm_backward(const Node *node) {
    // dx = s * (dy - x * dot(dy, x) * s^2 / d)
    const Tensor *a = node->in[0], *out = node->out;
    i64 cols = cols_of(a);
    f64 *g = grad_of(node->in[0]);
    for (i64 i = 0; g != NULL && i < rows_of(a); i++) {
        const f64 *x = &a->data[i * cols], *dy = &out->grad[i * cols];
        f64 s = rms_scale(x, cols, node->aux);
        f64 dot = 0.0;
        for (i64 j = 0; j < cols; j++) {
            dot += dy[j] * x[j];
        }
        for (i64 j = 0; j < cols; j++) {
            g[i * cols + j] += s * (dy[j] - x[j] * dot * s * s / (f64)cols);
        }
    }
}

Tensor *rms_norm(Tensor *a, f64 eps) {
    // no learned weight
    i64 cols = cols_of(a);
    Tensor *out = node_new(a->shape, a->ndim, rms_norm_backward, 1, (Tensor *[]){a}, eps, NULL);
    for (i64 i = 0; i < rows_of(a); i++) {
        f64 s = rms_scale(&a->data[i * cols], cols, eps);
        for (i64 j = 0; j < cols; j++) {
            out->data[i * cols + j] = a->data[i * cols + j] * s;
        }
    }
    return out;
}

static void softmax_row(const f64 *x, f64 *y, i64 cols) {
    // subtracts the max first, so large logits don't overflow exp
    f64 max = x[0];
    for (i64 j = 1; j < cols; j++) {
        max = x[j] > max ? x[j] : max;
    }
    f64 sum = 0.0;
    for (i64 j = 0; j < cols; j++) {
        y[j] = exp(x[j] - max);
        sum += y[j];
    }
    f64 inv = 1.0 / sum;
    for (i64 j = 0; j < cols; j++) {
        y[j] *= inv;
    }
}

static void softmax_backward(const Node *node) {
    // dx = y * (dy - dot(y, dy)), per row
    const Tensor *out = node->out;
    i64 cols = cols_of(out);
    f64 *g = grad_of(node->in[0]);
    for (i64 i = 0; g != NULL && i < rows_of(out); i++) {
        const f64 *y = &out->data[i * cols], *dy = &out->grad[i * cols];
        f64 dot = 0.0;
        for (i64 j = 0; j < cols; j++) {
            dot += y[j] * dy[j];
        }
        for (i64 j = 0; j < cols; j++) {
            g[i * cols + j] += y[j] * (dy[j] - dot);
        }
    }
}

Tensor *softmax(Tensor *a) {
    i64 cols = cols_of(a);
    Tensor *out = node_new(a->shape, a->ndim, softmax_backward, 1, (Tensor *[]){a}, 0, NULL);
    for (i64 i = 0; i < rows_of(a); i++) {
        softmax_row(&a->data[i * cols], &out->data[i * cols], cols);
    }
    return out;
}

static void cross_entropy_backward(const Node *node) {
    // dlogits = mask * (softmax(logits) - onehot(target)) / sum(mask)
    const Tensor *logits = node->in[0];
    i64 rows = rows_of(logits), cols = cols_of(logits);
    const i64 *targets = node->ctx;
    const f64 *mask = (const f64 *)(targets + rows);
    f64 *g = grad_of(node->in[0]);
    f64 sum_mask = 0.0;
    for (i64 i = 0; i < rows; i++) {
        sum_mask += mask[i];
    }
    f64 inv = sum_mask == 0.0 ? 0.0 : 1.0 / sum_mask;
    f64 *p = graph_alloc(cols * (i64)sizeof(f64));
    for (i64 i = 0; g != NULL && i < rows; i++) {
        softmax_row(&logits->data[i * cols], p, cols);
        for (i64 j = 0; j < cols; j++) {
            f64 onehot = j == targets[i] ? 1.0 : 0.0;
            g[i * cols + j] += node->out->grad[0] * (mask[i] * (p[j] - onehot) * inv);
        }
    }
}

Tensor *cross_entropy(Tensor *logits, const i64 *targets, const f64 *mask) {
    // fused with its softmax, loss = (1 / sum(mask)) * sum_i -mask[i] * log(p[i, target[i]])
    // ctx holds targets followed by mask in one buffer
    i64 rows = rows_of(logits), cols = cols_of(logits);
    i64 *ctx = graph_alloc(rows * (i64)(sizeof(i64) + sizeof(f64)));
    f64 *mask_copy = (f64 *)(ctx + rows);
    for (i64 i = 0; i < rows; i++) {
        assert(targets[i] >= 0 && targets[i] < cols && "cross_entropy: target out of range");
        ctx[i] = targets[i];
        mask_copy[i] = mask[i];
    }
    Tensor *out = node_new(NULL, 0, cross_entropy_backward, 1, (Tensor *[]){logits}, 0, ctx);
    f64 *p = graph_alloc(cols * (i64)sizeof(f64));
    f64 total = 0.0, sum_mask = 0.0;
    for (i64 i = 0; i < rows; i++) {
        softmax_row(&logits->data[i * cols], p, cols);
        total -= mask[i] * log(fmax(p[targets[i]], 1e-30));
        sum_mask += mask[i];
    }
    out->data[0] = sum_mask == 0.0 ? 0.0 : (1.0 / sum_mask) * total;
    return out;
}
