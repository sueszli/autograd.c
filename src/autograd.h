#pragma once

#include "tensor.h"

#define MAX_INPUTS 4

/*
 * every op appends a node to one global tape. creation order is a valid
 * topological order, so backward just walks the tape in reverse:
 *
 *   tape:      [h = x @ w1] [a = relu(h)] [loss = a @ w2]
 *   backward:                                  <- start here, seed 1
 *
 * the tape owns every op output. params are created with tensor_new and
 * owned by the caller. tape_reset frees all op outputs at once.
 */
typedef struct Node Node;
typedef void (*BackwardFn)(const Node *node);
struct Node {
    Tensor *out;
    Tensor *in[MAX_INPUTS];
    int64_t n_in;
    BackwardFn backward;
    void *ctx; // op-owned buffer, freed by tape_reset
};

// allocates an op output on the tape and records how to backprop into its inputs
Tensor *tape_push(int64_t rows, int64_t cols, BackwardFn backward, int64_t n_in, Tensor *const *in, void *ctx);
void tape_reset(void);

// lazily zero-initialized grad buffer, NULL if t does not require grad
double *grad_of(Tensor *t);

// seeds loss->grad with 1 and walks the tape in reverse. param grads accumulate across calls
void backward(Tensor *loss);
