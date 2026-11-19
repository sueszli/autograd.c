#include "tensor.h"
#include "std/linalg.h"
#include "std/mem.h"
#include <assert.h>
#include <math.h>
#include <string.h>

#define MAX_INPUTS 4
#define MAX_NODES 128 // one microGPT loss uses 54

// ===----------------------------------------------------------------------=== //
// memory
// ===----------------------------------------------------------------------=== //

static alignas(16) unsigned char perm_buf[512 * 1024];  // parity uses about 200 KB: two param sets, their grads and AdamW state
static alignas(16) unsigned char graph_buf[256 * 1024]; // one microGPT loss plus backward uses about 225 KB
static Arena perm = {.buf = perm_buf, .cap = sizeof(perm_buf)};
static Arena graph = {.buf = graph_buf, .cap = sizeof(graph_buf)};

void *perm_alloc(int64_t bytes) { return arena_alloc(&perm, (size_t)bytes); }
void perm_reset(void) { arena_reset(&perm); }
static void *graph_alloc(int64_t bytes) { return arena_alloc(&graph, (size_t)bytes); }

static void set_shape(Tensor *t, const int64_t *shape, int64_t ndim) {
    // copies the shape and multiplies it out into size
    assert(ndim >= 0 && ndim <= MAX_NDIM && "ndim out of range");
    t->ndim = ndim;
    t->size = 1;
    for (int64_t d = 0; d < ndim; d++) {
        assert(shape[d] > 0 && "tensor dims must be positive");
        t->shape[d] = shape[d];
        t->size *= shape[d];
    }
}

Tensor *tensor_new(const int64_t *shape, int64_t ndim, const double *data, bool requires_grad) {
    // params and inputs live in perm, so graph_reset never frees them
    Tensor *t = perm_alloc(sizeof(Tensor));
    set_shape(t, shape, ndim);
    int64_t bytes = t->size * (int64_t)sizeof(double);
    t->data = perm_alloc(bytes);
    t->grad = requires_grad ? perm_alloc(bytes) : NULL;
    t->requires_grad = requires_grad;
    if (data != NULL) {
        memcpy(t->data, data, (size_t)bytes);
    }
    return t;
}

// ===----------------------------------------------------------------------=== //
// autograd
// ===----------------------------------------------------------------------=== //

typedef struct Node Node;
typedef void (*BackwardFn)(const Node *node);

struct Node {
    Tensor *out;
    Tensor *in[MAX_INPUTS];
    int64_t n_in;
    BackwardFn backward;
    double aux; // scalar op argument, eg. a scale factor
    void *ctx;  // op scratch in graph
    bool visited;
};

static Tensor *node_new(const int64_t *shape, int64_t ndim, BackwardFn backward, int64_t n_in, Tensor *const *in, double aux, void *ctx) {
    // an op's node and its zeroed output, both in graph
    assert(n_in <= MAX_INPUTS && "too many inputs");
    Node *node = graph_alloc(sizeof(Node));
    *node = (Node){.n_in = n_in, .backward = backward, .aux = aux, .ctx = ctx};
    bool requires_grad = false;
    for (int64_t i = 0; i < n_in; i++) {
        node->in[i] = in[i];
        requires_grad |= in[i]->requires_grad;
    }
    Tensor *out = graph_alloc(sizeof(Tensor));
    set_shape(out, shape, ndim);
    out->data = graph_alloc(out->size * (int64_t)sizeof(double));
    out->node = node;
    out->requires_grad = requires_grad;
    node->out = out;
    return out;
}

static double *grad_of(Tensor *t) {
    // NULL if t does not require grad, op outputs get a buffer on first use
    if (!t->requires_grad) {
        return NULL;
    }
    if (t->grad == NULL) {
        t->grad = graph_alloc(t->size * (int64_t)sizeof(double));
    }
    return t->grad;
}

static Node *order[MAX_NODES];
static int64_t n_order;

static void topo_sort(const Tensor *t) {
    // postorder puts each node after all the nodes it depends on
    Node *node = t->node;
    if (!t->requires_grad || node == NULL || node->visited) {
        return;
    }
    node->visited = true;
    for (int64_t i = 0; i < node->n_in; i++) {
        topo_sort(node->in[i]);
    }
    assert(n_order < MAX_NODES && "too many nodes, raise MAX_NODES");
    order[n_order++] = node;
}

void backward(Tensor *loss) {
    // walks the sorted nodes from the loss down
    assert(loss->size == 1 && "loss must have one element");
    assert(loss->requires_grad && "loss does not depend on any param");
    assert(n_order == 0 && "backward called twice on one graph, call graph_reset in between");
    topo_sort(loss);
    grad_of(loss)[0] = 1.0;
    for (int64_t i = n_order - 1; i >= 0; i--) {
        order[i]->backward(order[i]);
    }
}

void graph_reset(void) {
    // frees every node, op output and grad of op outputs at once
    arena_reset(&graph);
    n_order = 0;
}

static int64_t cols_of(const Tensor *t) {
    // the last dim, which row-wise ops act on
    return t->shape[t->ndim - 1];
}

static int64_t rows_of(const Tensor *t) {
    // every leading dim flattened into rows
    return t->size / cols_of(t);
}

// ===----------------------------------------------------------------------=== //
// elementwise
// ===----------------------------------------------------------------------=== //

static void add_backward(const Node *node) {
    // both inputs get the output grad
    const Tensor *out = node->out;
    for (int64_t i = 0; i < node->n_in; i++) {
        double *g = grad_of(node->in[i]);
        for (int64_t j = 0; g != NULL && j < out->size; j++) {
            g[j] += out->grad[j];
        }
    }
}

Tensor *add(Tensor *a, Tensor *b) {
    assert(a->ndim == b->ndim && a->size == b->size && "add: shape mismatch");
    Tensor *out = node_new(a->shape, a->ndim, add_backward, 2, (Tensor *[]){a, b}, 0, NULL);
    for (int64_t i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return out;
}

static void mul_backward(const Node *node) {
    // d(a * s) / da = s
    const Tensor *out = node->out;
    double *g = grad_of(node->in[0]);
    for (int64_t i = 0; g != NULL && i < out->size; i++) {
        g[i] += node->aux * out->grad[i];
    }
}

Tensor *mul(Tensor *a, double s) {
    Tensor *out = node_new(a->shape, a->ndim, mul_backward, 1, (Tensor *[]){a}, s, NULL);
    for (int64_t i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] * s;
    }
    return out;
}

static void relu_backward(const Node *node) {
    // the grad passes where the input was positive
    const Tensor *a = node->in[0], *out = node->out;
    double *g = grad_of(node->in[0]);
    for (int64_t i = 0; g != NULL && i < out->size; i++) {
        g[i] += a->data[i] > 0.0 ? out->grad[i] : 0.0;
    }
}

Tensor *relu(Tensor *a) {
    Tensor *out = node_new(a->shape, a->ndim, relu_backward, 1, (Tensor *[]){a}, 0, NULL);
    for (int64_t i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] > 0.0 ? a->data[i] : 0.0;
    }
    return out;
}

// ===----------------------------------------------------------------------=== //
// matrix
// ===----------------------------------------------------------------------=== //

static void matmul_backward(const Node *node) {
    // ga += dout @ b^T, with b^T copied so the inner loop stays contiguous
    // gb += a^T @ dout
    const Tensor *a = node->in[0], *b = node->in[1], *out = node->out;
    int64_t n = rows_of(a), k = cols_of(a), m = cols_of(b);
    double *ga = grad_of(node->in[0]);
    double *gb = grad_of(node->in[1]);
    if (ga != NULL) {
        double *bt = graph_alloc(k * m * (int64_t)sizeof(double));
        linalg_transpose(bt, b->data, k, m);
        linalg_matmul(ga, out->grad, bt, n, m, k);
    }
    if (gb != NULL) {
        linalg_matmul_transposed_a(gb, a->data, out->grad, n, k, m);
    }
}

Tensor *matmul(Tensor *a, Tensor *b) {
    // leading dims of a fold into rows, so a batch is one big matmul
    assert(a->ndim >= 1 && b->ndim == 2 && cols_of(a) == b->shape[0] && "matmul: shape mismatch");
    int64_t shape[MAX_NDIM];
    memcpy(shape, a->shape, sizeof(shape));
    shape[a->ndim - 1] = b->shape[1];
    Tensor *out = node_new(shape, a->ndim, matmul_backward, 2, (Tensor *[]){a, b}, 0, NULL);
    linalg_matmul(out->data, a->data, b->data, rows_of(a), cols_of(a), b->shape[1]);
    return out;
}

static void transpose_backward(const Node *node) {
    // each batch matrix of the grad, transposed back
    const Tensor *out = node->out;
    int64_t r = out->shape[out->ndim - 2], c = cols_of(out);
    double *g = grad_of(node->in[0]);
    for (int64_t off = 0; g != NULL && off < out->size; off += r * c) {
        for (int64_t i = 0; i < r; i++) {
            for (int64_t j = 0; j < c; j++) {
                g[off + j * r + i] += out->grad[off + i * c + j];
            }
        }
    }
}

Tensor *transpose(Tensor *a) {
    // copies, there are no strided views
    assert(a->ndim >= 2 && "transpose: needs at least 2 dims");
    int64_t r = a->shape[a->ndim - 2], c = cols_of(a);
    int64_t shape[MAX_NDIM];
    memcpy(shape, a->shape, sizeof(shape));
    shape[a->ndim - 2] = c;
    shape[a->ndim - 1] = r;
    Tensor *out = node_new(shape, a->ndim, transpose_backward, 1, (Tensor *[]){a}, 0, NULL);
    for (int64_t off = 0; off < a->size; off += r * c) {
        linalg_transpose(&out->data[off], &a->data[off], r, c);
    }
    return out;
}

// ===----------------------------------------------------------------------=== //
// indexing
// ===----------------------------------------------------------------------=== //

static void embedding_backward(const Node *node) {
    // scatter rows back by id, summing duplicates
    const Tensor *out = node->out;
    const int64_t *ids = node->ctx;
    int64_t d = cols_of(out);
    double *g = grad_of(node->in[0]);
    for (int64_t r = 0; g != NULL && r < rows_of(out); r++) {
        for (int64_t c = 0; c < d; c++) {
            g[ids[r] * d + c] += out->grad[r * d + c];
        }
    }
}

Tensor *embedding(const int64_t *ids, int64_t n, Tensor *table) {
    // copies the ids, the caller's array may change before backward
    assert(n > 0 && table->ndim == 2 && "embedding: needs ids and a 2-d table");
    int64_t d = table->shape[1];
    int64_t *ids_copy = graph_alloc(n * (int64_t)sizeof(int64_t));
    for (int64_t r = 0; r < n; r++) {
        assert(ids[r] >= 0 && ids[r] < table->shape[0] && "embedding: id out of range");
        ids_copy[r] = ids[r];
    }
    Tensor *out = node_new(SHAPE(n, d), embedding_backward, 1, (Tensor *[]){table}, 0, ids_copy);
    for (int64_t r = 0; r < n; r++) {
        memcpy(&out->data[r * d], &table->data[ids[r] * d], (size_t)d * sizeof(double));
    }
    return out;
}

static void masked_fill_upper_backward(const Node *node) {
    // only the kept entries, on and below the diagonal, get a grad
    const Tensor *out = node->out;
    int64_t n = cols_of(out);
    double *g = grad_of(node->in[0]);
    for (int64_t off = 0; g != NULL && off < out->size; off += n * n) {
        for (int64_t i = 0; i < n; i++) {
            for (int64_t j = 0; j <= i; j++) {
                g[off + i * n + j] += out->grad[off + i * n + j];
            }
        }
    }
}

Tensor *masked_fill_upper(Tensor *a, double value) {
    // row i keeps columns j <= i, the causal mask of attention
    assert(a->ndim >= 2 && a->shape[a->ndim - 2] == cols_of(a) && "masked_fill_upper: last two dims must be square");
    int64_t n = cols_of(a);
    Tensor *out = node_new(a->shape, a->ndim, masked_fill_upper_backward, 1, (Tensor *[]){a}, 0, NULL);
    for (int64_t off = 0; off < a->size; off += n * n) {
        for (int64_t i = 0; i < n; i++) {
            for (int64_t j = 0; j < n; j++) {
                out->data[off + i * n + j] = j <= i ? a->data[off + i * n + j] : value;
            }
        }
    }
    return out;
}

static void narrow_backward(const Node *node) {
    // the slice's grad lands back in its columns
    const Tensor *a = node->in[0], *out = node->out;
    int64_t start = (int64_t)node->aux, len = cols_of(out), cols = cols_of(a);
    double *g = grad_of(node->in[0]);
    for (int64_t i = 0; g != NULL && i < rows_of(out); i++) {
        for (int64_t j = 0; j < len; j++) {
            g[i * cols + start + j] += out->grad[i * len + j];
        }
    }
}

Tensor *narrow(Tensor *a, int64_t start, int64_t len) {
    // splits attention heads apart
    int64_t cols = cols_of(a);
    assert(start >= 0 && len > 0 && start + len <= cols && "narrow: out of range");
    int64_t shape[MAX_NDIM];
    memcpy(shape, a->shape, sizeof(shape));
    shape[a->ndim - 1] = len;
    Tensor *out = node_new(shape, a->ndim, narrow_backward, 1, (Tensor *[]){a}, (double)start, NULL);
    for (int64_t i = 0; i < rows_of(a); i++) {
        memcpy(&out->data[i * len], &a->data[i * cols + start], (size_t)len * sizeof(double));
    }
    return out;
}

static void cat_backward(const Node *node) {
    // each part gets its own columns of the grad
    const Tensor *out = node->out;
    int64_t cols = cols_of(out);
    for (int64_t p = 0, start = 0; p < node->n_in; start += cols_of(node->in[p]), p++) {
        double *g = grad_of(node->in[p]);
        int64_t len = cols_of(node->in[p]);
        for (int64_t i = 0; g != NULL && i < rows_of(out); i++) {
            for (int64_t j = 0; j < len; j++) {
                g[i * len + j] += out->grad[i * cols + start + j];
            }
        }
    }
}

Tensor *cat(Tensor *const *parts, int64_t n) {
    // merges attention heads back together
    assert(n > 0 && "cat: no parts");
    int64_t cols = 0;
    for (int64_t p = 0; p < n; p++) {
        assert(parts[p]->ndim == parts[0]->ndim && rows_of(parts[p]) == rows_of(parts[0]) && "cat: leading dims mismatch");
        cols += cols_of(parts[p]);
    }
    int64_t shape[MAX_NDIM];
    memcpy(shape, parts[0]->shape, sizeof(shape));
    shape[parts[0]->ndim - 1] = cols;
    Tensor *out = node_new(shape, parts[0]->ndim, cat_backward, n, parts, 0, NULL);
    for (int64_t p = 0, start = 0; p < n; start += cols_of(parts[p]), p++) {
        int64_t len = cols_of(parts[p]);
        for (int64_t i = 0; i < rows_of(out); i++) {
            memcpy(&out->data[i * cols + start], &parts[p]->data[i * len], (size_t)len * sizeof(double));
        }
    }
    return out;
}

// ===----------------------------------------------------------------------=== //
// normalization and loss
// ===----------------------------------------------------------------------=== //

static double rms_scale(const double *row, int64_t cols, double eps) {
    // 1 / sqrt(mean(x^2) + eps), recomputed in backward so the same inputs give the same bits
    double sum_sq = 0.0;
    for (int64_t j = 0; j < cols; j++) {
        sum_sq += row[j] * row[j];
    }
    return pow(sum_sq * (1.0 / (double)cols) + eps, -0.5);
}

static void rms_norm_backward(const Node *node) {
    // dx = s * (dy - x * dot(dy, x) * s^2 / d)
    const Tensor *a = node->in[0], *out = node->out;
    int64_t cols = cols_of(a);
    double *g = grad_of(node->in[0]);
    for (int64_t i = 0; g != NULL && i < rows_of(a); i++) {
        const double *x = &a->data[i * cols], *dy = &out->grad[i * cols];
        double s = rms_scale(x, cols, node->aux);
        double dot = 0.0;
        for (int64_t j = 0; j < cols; j++) {
            dot += dy[j] * x[j];
        }
        for (int64_t j = 0; j < cols; j++) {
            g[i * cols + j] += s * (dy[j] - x[j] * dot * s * s / (double)cols);
        }
    }
}

Tensor *rms_norm(Tensor *a, double eps) {
    // no learned weight
    int64_t cols = cols_of(a);
    Tensor *out = node_new(a->shape, a->ndim, rms_norm_backward, 1, (Tensor *[]){a}, eps, NULL);
    for (int64_t i = 0; i < rows_of(a); i++) {
        double s = rms_scale(&a->data[i * cols], cols, eps);
        for (int64_t j = 0; j < cols; j++) {
            out->data[i * cols + j] = a->data[i * cols + j] * s;
        }
    }
    return out;
}

static void softmax_row(const double *x, double *y, int64_t cols) {
    // subtracts the max first, so large logits don't overflow exp
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

static void softmax_backward(const Node *node) {
    // dx = y * (dy - dot(y, dy)), per row
    const Tensor *out = node->out;
    int64_t cols = cols_of(out);
    double *g = grad_of(node->in[0]);
    for (int64_t i = 0; g != NULL && i < rows_of(out); i++) {
        const double *y = &out->data[i * cols], *dy = &out->grad[i * cols];
        double dot = 0.0;
        for (int64_t j = 0; j < cols; j++) {
            dot += y[j] * dy[j];
        }
        for (int64_t j = 0; j < cols; j++) {
            g[i * cols + j] += y[j] * (dy[j] - dot);
        }
    }
}

Tensor *softmax(Tensor *a) {
    int64_t cols = cols_of(a);
    Tensor *out = node_new(a->shape, a->ndim, softmax_backward, 1, (Tensor *[]){a}, 0, NULL);
    for (int64_t i = 0; i < rows_of(a); i++) {
        softmax_row(&a->data[i * cols], &out->data[i * cols], cols);
    }
    return out;
}

static void cross_entropy_backward(const Node *node) {
    // dlogits = mask * (softmax(logits) - onehot(target)) / sum(mask)
    const Tensor *logits = node->in[0];
    int64_t rows = rows_of(logits), cols = cols_of(logits);
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

Tensor *cross_entropy(Tensor *logits, const int64_t *targets, const double *mask) {
    // fused with its softmax, loss = (1 / sum(mask)) * sum_i -mask[i] * log(p[i, target[i]])
    // ctx holds targets followed by mask in one buffer
    int64_t rows = rows_of(logits), cols = cols_of(logits);
    int64_t *ctx = graph_alloc(rows * (int64_t)(sizeof(int64_t) + sizeof(double)));
    double *mask_copy = (double *)(ctx + rows);
    for (int64_t i = 0; i < rows; i++) {
        assert(targets[i] >= 0 && targets[i] < cols && "cross_entropy: target out of range");
        ctx[i] = targets[i];
        mask_copy[i] = mask[i];
    }
    Tensor *out = node_new(NULL, 0, cross_entropy_backward, 1, (Tensor *[]){logits}, 0, ctx);
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
