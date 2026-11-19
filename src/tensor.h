#pragma once

#include "std/mem.h"
#include "std/types.h"

static constexpr i64 MAX_NDIM = 4;

//
// n-d row-major f64 tensors with reverse-mode autograd, named after pytorch.
// https://pytorch.org/docs/stable/tensors.html
//
// data is contiguous, there are no strides and no broadcasting.
// row-wise ops act on the last dim.
// matmul, transpose and the mask act on the last two dims, and leading dims are a batch.
//
// every tensor points to the Mem it lives in, like a torch tensor points to its device.
// params live in mem->perm, and op outputs live in mem->graph.
// so ops take no context argument and the library keeps no global state.
//
// example:
// --------
//     static u8 perm_buf[1 << 16], graph_buf[1 << 16];
//     Mem mem = {.perm = {.buf = perm_buf, .cap = sizeof(perm_buf)}, .graph = {.buf = graph_buf, .cap = sizeof(graph_buf)}};
//     Tensor *w = tensor.new(&mem, shape(2, 2), (f64[]){1, 2, 3, 4}, true);
//     tensor.backward(tensor.cross_entropy(w, (i64[]){0, 1}, (f64[]){1, 1}));
//     tensor.reset_graph(&mem); // w->grad holds the gradient
//

typedef struct {
    Arena perm;  // params, their grads and optimizer state, never reset
    Arena graph; // nodes, op outputs, their grads and scratch, reset after each backward
    bool backward_done;
} Mem;

typedef struct Tensor {
    f64 *data;
    f64 *grad;         // params get one in tensor.new, op outputs when backward reaches them
    struct Node *node; // the op that made this tensor, NULL for params and inputs
    Mem *mem;          // where this tensor and everything computed from it lives
    i64 shape[MAX_NDIM];
    i64 ndim; // 0 for a scalar
    i64 size; // number of elements
    bool requires_grad;
} Tensor;

// a shape literal and its length, eg. tensor.new(&mem, shape(2, 3), NULL, true)
#define shape(...) (const i64[]){__VA_ARGS__}, (i64)(sizeof((i64[]){__VA_ARGS__}) / sizeof(i64))

typedef struct {
    // in mem->perm, data NULL gives zeros
    Tensor *(*new)(Mem *mem, const i64 *shape, i64 ndim, const f64 *data, bool requires_grad);

    // seeds loss->grad with 1 and backprops through the graph, param grads accumulate across calls
    void (*backward)(Tensor *loss);

    // frees every op output, call after each backward
    void (*reset_graph)(Mem *mem);

    // elementwise
    Tensor *(*add)(Tensor *a, Tensor *b); // a + b, same shape
    Tensor *(*mul)(Tensor *a, f64 s);     // a * s
    Tensor *(*relu)(Tensor *a);           // max(a, 0)

    // matrix
    Tensor *(*matmul)(Tensor *a, Tensor *b); // (..., n, k) @ (k, m) -> (..., n, m)
    Tensor *(*transpose)(Tensor *a);         // swaps the last two dims

    // indexing
    Tensor *(*embedding)(const i64 *ids, i64 n, Tensor *table); // rows of a 2-d table picked by ids, (n, d)
    Tensor *(*masked_fill_upper)(Tensor *a, f64 value);         // entries above the diagonal set to value, last two dims square
    Tensor *(*narrow)(Tensor *a, i64 start, i64 len);           // last dim [start, start + len)
    Tensor *(*cat)(Tensor *const *parts, i64 n);                // parts joined along the last dim

    // normalization and loss
    Tensor *(*rms_norm)(Tensor *a, f64 eps);                                       // each row divided by its root mean square
    Tensor *(*softmax)(Tensor *a);                                                 // softmax over the last dim
    Tensor *(*cross_entropy)(Tensor *logits, const i64 *targets, const f64 *mask); // scalar, mean over rows with mask 1
} TensorNamespace;

extern const TensorNamespace tensor;
