#pragma once

#include "std/types.h"
#include <stdbool.h>

#define MAX_NDIM 4

//
// n-d row-major float64 tensors with reverse-mode autograd, named after pytorch.
// https://pytorch.org/docs/stable/torch.html
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
// row-wise ops act on the last dim.
// matmul and transpose act on the last two dims, and leading dims are a batch.
// there is no broadcasting.
//
// all memory lives in two static arenas, and nothing calls malloc.
// perm holds params, their grads and optimizer state.
// graph holds nodes, op outputs, their grads and scratch.
//

typedef struct Tensor {
    f64 *data;
    f64 *grad;         // params get one in tensor_new, op outputs when backward reaches them
    struct Node *node; // the op that made this tensor, NULL for params and inputs
    i64 shape[MAX_NDIM];
    i64 ndim; // 0 for a scalar
    i64 size; // number of elements
    bool requires_grad;
} Tensor;

// a shape literal and its length, eg. tensor_new(SHAPE(2, 3), NULL, true)
#define SHAPE(...) (const i64[]){__VA_ARGS__}, (i64)(sizeof((i64[]){__VA_ARGS__}) / sizeof(i64))

// in perm, data NULL gives zeros
Tensor *tensor_new(const i64 *shape, i64 ndim, const f64 *data, bool requires_grad);

// ===----------------------------------------------------------------------=== //
// autograd and memory
// ===----------------------------------------------------------------------=== //

// seeds loss->grad with 1 and backprops through the graph
// param grads accumulate across calls
void backward(Tensor *loss);

void graph_reset(void);      // frees the graph, call after each backward
void *perm_alloc(i64 bytes); // zeroed
void perm_reset(void);       // frees every param, only tests need this

// ===----------------------------------------------------------------------=== //
// ops
// ===----------------------------------------------------------------------=== //

// elementwise
Tensor *add(Tensor *a, Tensor *b); // a + b, same shape
Tensor *mul(Tensor *a, f64 s);     // a * s
Tensor *relu(Tensor *a);           // max(a, 0)

// matrix
Tensor *matmul(Tensor *a, Tensor *b); // (..., n, k) @ (k, m) -> (..., n, m)
Tensor *transpose(Tensor *a);         // swaps the last two dims

// indexing
Tensor *embedding(const i64 *ids, i64 n, Tensor *table); // rows of a 2-d table picked by ids, (n, d)
Tensor *masked_fill_upper(Tensor *a, f64 value);         // entries above the diagonal set to value, last two dims square
Tensor *narrow(Tensor *a, i64 start, i64 len);           // last dim [start, start + len)
Tensor *cat(Tensor *const *parts, i64 n);                // parts joined along the last dim

// normalization and loss
Tensor *rms_norm(Tensor *a, f64 eps);                                       // each row divided by its root mean square
Tensor *softmax(Tensor *a);                                                 // softmax over the last dim
Tensor *cross_entropy(Tensor *logits, const i64 *targets, const f64 *mask); // scalar, mean over rows with mask 1
