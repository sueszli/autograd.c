#pragma once

#include "tensor.h"

static constexpr i64 MAX_INPUTS = 4;

//
// internals for ops, users only need tensor.h.
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

// an op's node and its zeroed output, both in the graph of in[0]
[[nodiscard]] Tensor *node_new(const i64 *shape, i64 ndim, BackwardFn backward, i64 n_in, Tensor *const *in, f64 aux, void *ctx);

// grad buffer, NULL if t does not require grad
// op outputs get one on first use
[[nodiscard]] f64 *grad_of(Tensor *t);

[[nodiscard]] void *graph_alloc(Mem *mem, i64 bytes); // zeroed

// copies the shape and multiplies it out into size
void set_shape(Tensor *t, const i64 *shape, i64 ndim);

// the last dim, which row-wise ops act on
static inline i64 cols_of(const Tensor *t) { return t->shape[t->ndim - 1]; }

// every leading dim flattened into rows
static inline i64 rows_of(const Tensor *t) { return t->size / cols_of(t); }

void autograd_backward(Tensor *loss);
void autograd_reset(Mem *mem);
