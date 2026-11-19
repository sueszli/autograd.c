#include "tensor.h"
#include <assert.h>
#include <math.h>
#include <string.h>

static constexpr i64 MAX_INPUTS = 4;
static constexpr i64 MAX_NODES = 128; // one microGPT loss uses 54

// ===----------------------------------------------------------------------=== //
// shapes and memory
// ===----------------------------------------------------------------------=== //

static void set_shape(Tensor *t, const i64 *shape, i64 ndim) {
    // copies the shape and multiplies it out into size
    assert(ndim >= 0 && ndim <= MAX_NDIM && "ndim out of range");
    t->ndim = ndim;
    t->size = 1;
    for (i64 d = 0; d < ndim; d++) {
        assert(shape[d] > 0 && "tensor dims must be positive");
        t->shape[d] = shape[d];
        t->size *= shape[d];
    }
}

static Tensor *tensor_new(Mem *mem, const i64 *shape, i64 ndim, const f64 *data, bool requires_grad) {
    // params and inputs live in perm, so reset_graph never frees them
    Tensor *t = arena.alloc(&mem->perm, sizeof(Tensor));
    set_shape(t, shape, ndim);
    usize bytes = (usize)t->size * sizeof(f64);
    t->data = arena.alloc(&mem->perm, bytes);
    t->grad = requires_grad ? arena.alloc(&mem->perm, bytes) : NULL;
    t->mem = mem;
    t->requires_grad = requires_grad;
    if (data != NULL) {
        memcpy(t->data, data, bytes);
    }
    return t;
}

static i64 cols_of(const Tensor *t) {
    // the last dim, which row-wise ops act on
    return t->shape[t->ndim - 1];
}

static i64 rows_of(const Tensor *t) {
    // every leading dim flattened into rows
    return t->size / cols_of(t);
}

// ===----------------------------------------------------------------------=== //
// autograd
// ===----------------------------------------------------------------------=== //

//
// every op output points to the node that made it.
// every node points to its inputs.
//
//     x  ─┐
//     w1 ─┴─ [matmul] ─ h ─ [relu] ─ a ─┐
//     w2 ───────────────────────────────┴─ [matmul] ─ loss
//
// backward sorts the nodes behind the loss topologically with a dfs.
// then it runs each node's backward from the loss down to the params.
//

typedef struct Node Node;
typedef void (*BackwardFn)(const Node *node);

struct Node {
    Tensor *out;
    Tensor *in[MAX_INPUTS];
    i64 n_in;
    BackwardFn backward;
    f64 aux;   // scalar op argument, eg. a scale factor
    void *ctx; // op scratch in graph
    bool visited;
};

// backward passes, each defined right below its forward op
static void add_backward(const Node *node);
static void mul_backward(const Node *node);
static void relu_backward(const Node *node);
static void matmul_backward(const Node *node);
static void transpose_backward(const Node *node);
static void embedding_backward(const Node *node);
static void masked_fill_upper_backward(const Node *node);
static void narrow_backward(const Node *node);
static void cat_backward(const Node *node);
static void rms_norm_backward(const Node *node);
static void softmax_backward(const Node *node);
static void cross_entropy_backward(const Node *node);

static void *graph_alloc(Mem *mem, i64 bytes) { return arena.alloc(&mem->graph, (usize)bytes); }

static Tensor *node_new(const i64 *shape, i64 ndim, BackwardFn backward, i64 n_in, Tensor *const *in, f64 aux, void *ctx) {
    // an op's node and its zeroed output, both in the graph of in[0]
    assert(n_in > 0 && n_in <= MAX_INPUTS && "op needs 1 to MAX_INPUTS inputs");
    Mem *mem = in[0]->mem;
    Node *node = graph_alloc(mem, sizeof(Node));
    *node = (Node){.n_in = n_in, .backward = backward, .aux = aux, .ctx = ctx};
    bool requires_grad = false;
    for (i64 i = 0; i < n_in; i++) {
        assert(in[i]->mem == mem && "op inputs must share one Mem");
        node->in[i] = in[i];
        requires_grad |= in[i]->requires_grad;
    }
    Tensor *out = graph_alloc(mem, sizeof(Tensor));
    set_shape(out, shape, ndim);
    out->data = graph_alloc(mem, out->size * (i64)sizeof(f64));
    out->node = node;
    out->mem = mem;
    out->requires_grad = requires_grad;
    node->out = out;
    return out;
}

static f64 *grad_of(Tensor *t) {
    // NULL if t does not require grad, op outputs get a buffer on first use
    if (!t->requires_grad) {
        return NULL;
    }
    if (t->grad == NULL) {
        t->grad = graph_alloc(t->mem, t->size * (i64)sizeof(f64));
    }
    return t->grad;
}

static void topo_sort(const Tensor *t, Node **order, i64 *n) {
    // postorder puts each node after all the nodes it depends on
    Node *node = t->node;
    if (!t->requires_grad || node == NULL || node->visited) {
        return;
    }
    node->visited = true;
    for (i64 i = 0; i < node->n_in; i++) {
        topo_sort(node->in[i], order, n);
    }
    assert(*n < MAX_NODES && "too many nodes, raise MAX_NODES");
    order[(*n)++] = node;
}

static void autograd_backward(Tensor *loss) {
    // walks the sorted nodes from the loss down, the order lives in the graph too
    Mem *mem = loss->mem;
    assert(loss->size == 1 && "loss must have one element");
    assert(loss->requires_grad && "loss does not depend on any param");
    assert(!mem->backward_done && "backward called twice on one graph, call reset_graph in between");
    mem->backward_done = true;
    Node **order = graph_alloc(mem, MAX_NODES * (i64)sizeof(Node *));
    i64 n = 0;
    topo_sort(loss, order, &n);
    grad_of(loss)[0] = 1.0;
    for (i64 i = n - 1; i >= 0; i--) {
        order[i]->backward(order[i]);
    }
}

static void autograd_reset(Mem *mem) {
    // frees every node, op output and grad of op outputs at once
    arena.reset(&mem->graph);
    mem->backward_done = false;
}

// ===----------------------------------------------------------------------=== //
// elementwise
// ===----------------------------------------------------------------------=== //

static Tensor *op_add(Tensor *a, Tensor *b) {
    assert(a->ndim == b->ndim && a->size == b->size && "add: shape mismatch");
    Tensor *out = node_new(a->shape, a->ndim, add_backward, 2, (Tensor *[]){a, b}, 0, NULL);
    for (i64 i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return out;
}

static void add_backward(const Node *node) {
    // both inputs get the output grad
    const Tensor *out = node->out;
    for (i64 i = 0; i < node->n_in; i++) {
        f64 *g = grad_of(node->in[i]);
        for (i64 j = 0; g != NULL && j < out->size; j++) {
            g[j] += out->grad[j];
        }
    }
}

static Tensor *op_mul(Tensor *a, f64 s) {
    Tensor *out = node_new(a->shape, a->ndim, mul_backward, 1, (Tensor *[]){a}, s, NULL);
    for (i64 i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] * s;
    }
    return out;
}

static void mul_backward(const Node *node) {
    // d(a * s) / da = s
    const Tensor *out = node->out;
    f64 *g = grad_of(node->in[0]);
    for (i64 i = 0; g != NULL && i < out->size; i++) {
        g[i] += node->aux * out->grad[i];
    }
}

static Tensor *op_relu(Tensor *a) {
    Tensor *out = node_new(a->shape, a->ndim, relu_backward, 1, (Tensor *[]){a}, 0, NULL);
    for (i64 i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] > 0.0 ? a->data[i] : 0.0;
    }
    return out;
}

static void relu_backward(const Node *node) {
    // the grad passes where the input was positive
    const Tensor *a = node->in[0], *out = node->out;
    f64 *g = grad_of(node->in[0]);
    for (i64 i = 0; g != NULL && i < out->size; i++) {
        g[i] += a->data[i] > 0.0 ? out->grad[i] : 0.0;
    }
}

// ===----------------------------------------------------------------------=== //
// matrix
// ===----------------------------------------------------------------------=== //

// the kernels sum every output element in ascending order, so results are reproducible
// restrict promises out never overlaps a or b, so the compiler can vectorize

static void kernel_matmul(f64 *restrict out, const f64 *restrict a, const f64 *restrict b, i64 n, i64 k, i64 m) {
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

static void kernel_matmul_transposed_a(f64 *restrict out, const f64 *restrict a, const f64 *restrict b, i64 n, i64 k, i64 m) {
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

static void kernel_transpose(f64 *restrict out, const f64 *restrict a, i64 n, i64 m) {
    // out (m x n) = a^T, for a (n x m)
    for (i64 i = 0; i < n; i++) {
        for (i64 j = 0; j < m; j++) {
            out[j * n + i] = a[i * m + j];
        }
    }
}

static Tensor *op_matmul(Tensor *a, Tensor *b) {
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
        f64 *bt = graph_alloc(out->mem, k * m * (i64)sizeof(f64));
        kernel_transpose(bt, b->data, k, m);
        kernel_matmul(ga, out->grad, bt, n, m, k);
    }
    if (gb != NULL) {
        kernel_matmul_transposed_a(gb, a->data, out->grad, n, k, m);
    }
}

static Tensor *op_transpose(Tensor *a) {
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

// ===----------------------------------------------------------------------=== //
// indexing
// ===----------------------------------------------------------------------=== //

static Tensor *op_embedding(const i64 *ids, i64 n, Tensor *table) {
    // copies the ids, the caller's array may change before backward
    assert(n > 0 && table->ndim == 2 && "embedding: needs ids and a 2-d table");
    i64 d = table->shape[1];
    i64 *ids_copy = graph_alloc(table->mem, n * (i64)sizeof(i64));
    for (i64 r = 0; r < n; r++) {
        assert(ids[r] >= 0 && ids[r] < table->shape[0] && "embedding: id out of range");
        ids_copy[r] = ids[r];
    }
    Tensor *out = node_new(shape(n, d), embedding_backward, 1, (Tensor *[]){table}, 0, ids_copy);
    for (i64 r = 0; r < n; r++) {
        memcpy(&out->data[r * d], &table->data[ids[r] * d], (usize)d * sizeof(f64));
    }
    return out;
}

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

static Tensor *op_masked_fill_upper(Tensor *a, f64 value) {
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

static Tensor *op_narrow(Tensor *a, i64 start, i64 len) {
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

static Tensor *op_cat(Tensor *const *parts, i64 n) {
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

// ===----------------------------------------------------------------------=== //
// normalization and loss
// ===----------------------------------------------------------------------=== //

static f64 rms_scale(const f64 *row, i64 cols, f64 eps) {
    // 1 / sqrt(mean(x^2) + eps), recomputed in backward so the same inputs give the same bits
    f64 sum_sq = 0.0;
    for (i64 j = 0; j < cols; j++) {
        sum_sq += row[j] * row[j];
    }
    return pow(sum_sq * (1.0 / (f64)cols) + eps, -0.5);
}

static Tensor *op_rms_norm(Tensor *a, f64 eps) {
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

static Tensor *op_softmax(Tensor *a) {
    i64 cols = cols_of(a);
    Tensor *out = node_new(a->shape, a->ndim, softmax_backward, 1, (Tensor *[]){a}, 0, NULL);
    for (i64 i = 0; i < rows_of(a); i++) {
        softmax_row(&a->data[i * cols], &out->data[i * cols], cols);
    }
    return out;
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

static Tensor *op_cross_entropy(Tensor *logits, const i64 *targets, const f64 *mask) {
    // fused with its softmax, loss = (1 / sum(mask)) * sum_i -mask[i] * log(p[i, target[i]])
    // ctx holds targets followed by mask in one buffer
    i64 rows = rows_of(logits), cols = cols_of(logits);
    i64 *ctx = graph_alloc(logits->mem, rows * (i64)(sizeof(i64) + sizeof(f64)));
    f64 *mask_copy = (f64 *)(ctx + rows);
    for (i64 i = 0; i < rows; i++) {
        assert(targets[i] >= 0 && targets[i] < cols && "cross_entropy: target out of range");
        ctx[i] = targets[i];
        mask_copy[i] = mask[i];
    }
    Tensor *out = node_new(NULL, 0, cross_entropy_backward, 1, (Tensor *[]){logits}, 0, ctx);
    f64 *p = graph_alloc(logits->mem, cols * (i64)sizeof(f64));
    f64 total = 0.0, sum_mask = 0.0;
    for (i64 i = 0; i < rows; i++) {
        softmax_row(&logits->data[i * cols], p, cols);
        total -= mask[i] * log(fmax(p[targets[i]], 1e-30));
        sum_mask += mask[i];
    }
    out->data[0] = sum_mask == 0.0 ? 0.0 : (1.0 / sum_mask) * total;
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
    f64 *p = graph_alloc(node->out->mem, cols * (i64)sizeof(f64));
    for (i64 i = 0; g != NULL && i < rows; i++) {
        softmax_row(&logits->data[i * cols], p, cols);
        for (i64 j = 0; j < cols; j++) {
            f64 onehot = j == targets[i] ? 1.0 : 0.0;
            g[i * cols + j] += node->out->grad[0] * (mask[i] * (p[j] - onehot) * inv);
        }
    }
}

// ===----------------------------------------------------------------------=== //
// namespace
// ===----------------------------------------------------------------------=== //

const TensorNamespace tensor = {
    .new = tensor_new,
    .backward = autograd_backward,
    .reset_graph = autograd_reset,
    .add = op_add,
    .mul = op_mul,
    .relu = op_relu,
    .matmul = op_matmul,
    .transpose = op_transpose,
    .embedding = op_embedding,
    .masked_fill_upper = op_masked_fill_upper,
    .narrow = op_narrow,
    .cat = op_cat,
    .rms_norm = op_rms_norm,
    .softmax = op_softmax,
    .cross_entropy = op_cross_entropy,
};
