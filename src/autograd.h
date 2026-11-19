#pragma once

#include "tensor.h"

#define MAX_INPUTS 4
#define TAPE_CAP 128             // nodes, one microGPT loss plus backward at BLOCK_SIZE uses 54
#define ARENA_BYTES (256 * 1024) // op outputs, their grads and ctx, one microGPT loss plus backward uses 184032

/*
 * every op appends a node to one global tape. creation order is a valid
 * topological order, so backward just walks the tape in reverse:
 *
 *   tape:      [h = x @ w1] [a = relu(h)] [loss = a @ w2]
 *   backward:                                  <- start here, seed 1
 *
 * the tape owns every op output. their data, grads and ctx live in one static
 * arena, so nothing is malloc'd after startup. params are created with
 * tensor_new and owned by the caller. tape_reset rewinds the tape and arena.
 */
typedef struct Node Node;
typedef void (*BackwardFn)(const Node *node);
struct Node {
    Tensor *out;
    Tensor *in[MAX_INPUTS];
    int64_t n_in;
    BackwardFn backward;
    double aux; // scalar op argument, eg. a scale factor
    void *ctx;  // op-owned buffer from tape_alloc
};

// allocates an op output on the tape and records how to backprop into its inputs
Tensor *tape_push(int64_t rows, int64_t cols, BackwardFn backward, int64_t n_in, Tensor *const *in, double aux, void *ctx);
void tape_reset(void);

// zeroed memory from the arena, valid until the next tape_reset
void *tape_alloc(int64_t bytes);

// grad buffer, NULL if t does not require grad. params get one in tensor_new, op outputs get one from the arena on first use
double *grad_of(Tensor *t);

// seeds loss->grad with 1 and walks the tape in reverse. param grads accumulate across calls
void backward(Tensor *loss);
