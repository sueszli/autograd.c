#pragma once

#include "alloc.h"
#include "tensor.h"

#define MAX_INPUTS 4
#define MAX_NODES 128 // one microGPT loss uses 54

/*
 * every op output points to the node that made it, every node points to its inputs:
 *
 *   x  ─┐
 *   w1 ─┴─ [matmul] ─ h ─ [relu] ─ a ─┐
 *   w2 ───────────────────────────────┴─ [matmul] ─ loss
 *
 * backward sorts the nodes behind the loss topologically with a dfs,
 * then runs each node's backward from the loss down to the params.
 */
typedef struct Node Node;
typedef void (*BackwardFn)(const Node *node);
struct Node {
    Tensor *out;
    Tensor *in[MAX_INPUTS];
    int64_t n_in;
    BackwardFn backward;
    double aux; // scalar op argument, eg. a scale factor
    void *ctx;  // op scratch in graph_mem
    bool visited;
};

// creates an op's node and its output tensor in graph_mem
Tensor *node_new(int64_t rows, int64_t cols, BackwardFn backward, int64_t n_in, Tensor *const *in, double aux, void *ctx);

// frees the graph, call after each backward
void graph_reset(void);

// grad buffer, NULL if t does not require grad. op outputs get one on first use
double *grad_of(Tensor *t);

// seeds loss->grad with 1 and backprops through the graph. param grads accumulate across calls
void backward(Tensor *loss);
