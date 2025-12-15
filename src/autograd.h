#pragma once

#include <stddef.h>
#include <stdint.h>

#define MAX_INPUTS 4

// avoid circular dependency
typedef struct Tensor Tensor;

/*
 * represents an op in the computation graph
 *
 * example:
 *    forward graph:  (a, b) -> mul -> result -> sum -> loss
 *    backward flow:  loss -> sum -> mul -> (a, b)
 *
 * forward:
 *    Tensor *result = mul(a, b);
 *    if (result->requires_grad) {
 *        Function *fn = malloc_arena();
 *        fn->apply = mul_backward;                       // callback to compute gradients on backprop
 *        fn->num_inputs = 2;
 *        fn->inputs[0] = a;
 *        fn->inputs[1] = b;
 *        fn->output = result;
 *        fn->pending_count = 0;                          // how many ops must finish before this one can run
 *        if (a->grad_fn) a->grad_fn->pending_count++;
 *        if (b->grad_fn) b->grad_fn->pending_count++;
 *        fn->ctx = NULL;                                 // no extra storage needed
 *        result->grad_fn = fn;
 *    }
 *
 * backward:
 *    after forward pass in neural net, traverse graph backward from loss
 *    calling each Function's apply() to compute gradients
 *    to find the contribution of each operation to the final loss.
 *
 *    results can then be retrieved from each Tensor's grad field.
 */
typedef struct Function {
    void (*apply)(struct Function *self, const Tensor *grad_output); // callback to compute gradients
    uint32_t num_inputs;                                             // number of args for op
    Tensor *inputs[MAX_INPUTS];                                      // args
    Tensor *output;                                                  // result Tensor
    uint32_t pending_count;                                          // how many ops must finish before this one can run
    void *ctx;                                                       // extra metadata storage, used by complex operations
} Function;

// memory arena to store Function structs for all ops in a thread
typedef struct Arena {
    void *memory;
    size_t capacity;
    size_t offset;
} Arena;
Function *arena_alloc_function(void);
void arena_free(void);

// traverse computation graph backward from loss tensor
void backward(Tensor *loss);

// add a new gradient to tensor->grad
void accumulate_grad(Tensor *tensor, Tensor *new_grad);
