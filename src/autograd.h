#pragma once

#include "tensor.h"

enum { max_inputs = 4 };

//
// reverse-mode autograd over a graph of op nodes.
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
// nodes, op outputs, their grads and scratch live in the graph arena.
//

// ===----------------------------------------------------------------------=== //
// for users
// ===----------------------------------------------------------------------=== //

// seeds loss->grad with 1 and backprops through the graph
// param grads accumulate across calls
void backward(Tensor *loss);

// frees the graph, call after each backward
void graph_reset(void);

// ===----------------------------------------------------------------------=== //
// for ops
// ===----------------------------------------------------------------------=== //

typedef struct Node Node;
typedef void (*BackwardFn)(const Node *node);

struct Node {
    Tensor *out;
    Tensor *in[max_inputs];
    i64 n_in;
    BackwardFn backward;
    f64 aux;   // scalar op argument, eg. a scale factor
    void *ctx; // op scratch in graph
    bool visited;
};

// an op's node and its zeroed output, both in graph
Tensor *node_new(const i64 *shape, i64 ndim, BackwardFn backward, i64 n_in, Tensor *const *in, f64 aux, void *ctx);

// grad buffer, NULL if t does not require grad
// op outputs get one on first use
f64 *grad_of(Tensor *t);

void *graph_alloc(i64 bytes); // zeroed
