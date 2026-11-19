#pragma once

#include "stack.h"
#include "tensor.h"

#define MAX_INPUTS 4

/*
 * every op pushes a node onto the nodes stack. push order is a valid
 * topological order, so backward walks the stack from the top down:
 *
 *   nodes:     [h = x @ w1] [a = relu(h)] [loss = a @ w2]
 *   backward:                                  <- start here, seed 1
 *
 * op outputs, their grads and scratch buffers live on the mem stack.
 * params come from tensor_new and are owned by the caller.
 */
typedef struct Node Node;
typedef void (*BackwardFn)(const Node *node);
struct Node {
    Tensor *out;
    Tensor *in[MAX_INPUTS];
    int64_t n_in;
    BackwardFn backward;
    double aux; // scalar op argument, eg. a scale factor
    void *ctx;  // op scratch on the mem stack
};

extern Stack nodes; // 128 nodes, one microGPT loss uses 54
extern Stack mem;   // 256 KiB, one microGPT loss plus backward uses about 184 KB

// pushes an op output and the node that backprops into its inputs
Tensor *node_push(int64_t rows, int64_t cols, BackwardFn backward, int64_t n_in, Tensor *const *in, double aux, void *ctx);

// empties both stacks, call after each backward
void autograd_reset(void);

// grad buffer, NULL if t does not require grad. op outputs get one on first use
double *grad_of(Tensor *t);

// seeds loss->grad with 1 and walks the nodes from the top down. param grads accumulate across calls
void backward(Tensor *loss);
