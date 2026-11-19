#include "autograd.h"
#include "ops.h"
#include <assert.h>
#include <string.h>

static void embedding_backward(const Node *node) {
    // scatter rows back by id, summing duplicates
    const Tensor *out = node->out;
    const i64 *ids = node->ctx;
    i64 d = cols_of(out);
    f64 *g = grad_of(node->in[0]);
    for (i64 r = 0; g != NULL && r < rows_of(out); r++) {
        for (i64 c = 0; c < d; c++) {
            g[ids[r] * d + c] += out->grad[r * d + c];
        }
    }
}

Tensor *embedding(const i64 *ids, i64 n, Tensor *table) {
    // copies the ids, the caller's array may change before backward
    assert(n > 0 && table->ndim == 2 && "embedding: needs ids and a 2-d table");
    i64 d = table->shape[1];
    i64 *ids_copy = graph_alloc(n * (i64)sizeof(i64));
    for (i64 r = 0; r < n; r++) {
        assert(ids[r] >= 0 && ids[r] < table->shape[0] && "embedding: id out of range");
        ids_copy[r] = ids[r];
    }
    Tensor *out = node_new(SHAPE(n, d), embedding_backward, 1, (Tensor *[]){table}, 0, ids_copy);
    for (i64 r = 0; r < n; r++) {
        memcpy(&out->data[r * d], &table->data[ids[r] * d], (usize)d * sizeof(f64));
    }
    return out;
}

static void masked_fill_upper_backward(const Node *node) {
    // only the kept entries, on and below the diagonal, get a grad
    const Tensor *out = node->out;
    i64 n = cols_of(out);
    f64 *g = grad_of(node->in[0]);
    for (i64 off = 0; g != NULL && off < out->size; off += n * n) {
        for (i64 i = 0; i < n; i++) {
            for (i64 j = 0; j <= i; j++) {
                g[off + i * n + j] += out->grad[off + i * n + j];
            }
        }
    }
}

Tensor *masked_fill_upper(Tensor *a, f64 value) {
    // row i keeps columns j <= i, the causal mask of attention
    assert(a->ndim >= 2 && a->shape[a->ndim - 2] == cols_of(a) && "masked_fill_upper: last two dims must be square");
    i64 n = cols_of(a);
    Tensor *out = node_new(a->shape, a->ndim, masked_fill_upper_backward, 1, (Tensor *[]){a}, 0, NULL);
    for (i64 off = 0; off < a->size; off += n * n) {
        for (i64 i = 0; i < n; i++) {
            for (i64 j = 0; j < n; j++) {
                out->data[off + i * n + j] = j <= i ? a->data[off + i * n + j] : value;
            }
        }
    }
    return out;
}

static void narrow_backward(const Node *node) {
    // the slice's grad lands back in its columns
    const Tensor *a = node->in[0], *out = node->out;
    i64 start = (i64)node->aux, len = cols_of(out), cols = cols_of(a);
    f64 *g = grad_of(node->in[0]);
    for (i64 i = 0; g != NULL && i < rows_of(out); i++) {
        for (i64 j = 0; j < len; j++) {
            g[i * cols + start + j] += out->grad[i * len + j];
        }
    }
}

Tensor *narrow(Tensor *a, i64 start, i64 len) {
    // splits attention heads apart
    i64 cols = cols_of(a);
    assert(start >= 0 && len > 0 && start + len <= cols && "narrow: out of range");
    i64 shape[MAX_NDIM];
    memcpy(shape, a->shape, sizeof(shape));
    shape[a->ndim - 1] = len;
    Tensor *out = node_new(shape, a->ndim, narrow_backward, 1, (Tensor *[]){a}, (f64)start, NULL);
    for (i64 i = 0; i < rows_of(a); i++) {
        memcpy(&out->data[i * len], &a->data[i * cols + start], (usize)len * sizeof(f64));
    }
    return out;
}

static void cat_backward(const Node *node) {
    // each part gets its own columns of the grad
    const Tensor *out = node->out;
    i64 cols = cols_of(out);
    for (i64 p = 0, start = 0; p < node->n_in; start += cols_of(node->in[p]), p++) {
        f64 *g = grad_of(node->in[p]);
        i64 len = cols_of(node->in[p]);
        for (i64 i = 0; g != NULL && i < rows_of(out); i++) {
            for (i64 j = 0; j < len; j++) {
                g[i * len + j] += out->grad[i * cols + start + j];
            }
        }
    }
}

Tensor *cat(Tensor *const *parts, i64 n) {
    // merges attention heads back together
    assert(n > 0 && "cat: no parts");
    i64 cols = 0;
    for (i64 p = 0; p < n; p++) {
        assert(parts[p]->ndim == parts[0]->ndim && rows_of(parts[p]) == rows_of(parts[0]) && "cat: leading dims mismatch");
        cols += cols_of(parts[p]);
    }
    i64 shape[MAX_NDIM];
    memcpy(shape, parts[0]->shape, sizeof(shape));
    shape[parts[0]->ndim - 1] = cols;
    Tensor *out = node_new(shape, parts[0]->ndim, cat_backward, n, parts, 0, NULL);
    for (i64 p = 0, start = 0; p < n; start += cols_of(parts[p]), p++) {
        i64 len = cols_of(parts[p]);
        for (i64 i = 0; i < rows_of(out); i++) {
            memcpy(&out->data[i * cols + start], &parts[p]->data[i * len], (usize)len * sizeof(f64));
        }
    }
    return out;
}
