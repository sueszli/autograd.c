#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define MAX_INPUTS 4
#define MAX_NODES 128 // one microGPT loss uses 54

// ===----------------------------------------------------------------------=== //
// for users
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
// all memory lives in two static arenas, and nothing calls malloc.
// perm holds params, their grads and optimizer state.
// graph holds nodes, op outputs, their grads and scratch.
//

typedef struct Node Node;

// 2-D row-major float64, the only shape microGPT needs
typedef struct {
    double *data;
    double *grad; // params get one in tensor_new, op outputs when backward reaches them
    Node *node;   // the op that made this tensor, NULL for params and inputs
    int64_t rows;
    int64_t cols;
    bool requires_grad;
} Tensor;

Tensor *tensor_new(int64_t rows, int64_t cols, const double *data, bool requires_grad); // in perm, data NULL gives zeros

// seeds loss->grad with 1 and backprops through the graph
// param grads accumulate across calls
void backward(Tensor *loss);

void *perm_alloc(int64_t bytes);  // zeroed
void perm_reset(void);            // frees every param, only tests need this
void *graph_alloc(int64_t bytes); // zeroed
void graph_reset(void);           // frees the graph, call after each backward

// ===----------------------------------------------------------------------=== //
// for ops
// ===----------------------------------------------------------------------=== //

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

// creates an op's node and its output tensor in graph
Tensor *node_new(int64_t rows, int64_t cols, BackwardFn backward, int64_t n_in, Tensor *const *in, double aux, void *ctx);

// grad buffer, NULL if t does not require grad
// op outputs get one on first use
double *grad_of(Tensor *t);
