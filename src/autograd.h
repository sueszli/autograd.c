#pragma once

#include <stddef.h>
#include <stdint.h>

#define MAX_INPUTS 4

// avoid circular dependency
typedef struct Tensor Tensor;

// represents a backward operation in the computation graph
typedef struct Function {
    void (*apply)(struct Function *self, const Tensor *grad_output); // backward kernel
    Tensor *output;                                                  // tensor this function produced
    Tensor *inputs[MAX_INPUTS];                                      // parent tensors (fixed-size array)
    uint32_t num_inputs;                                             // actual number of inputs
    uint32_t pending_count;                                          // downstream consumer edges
    void *ctx;                                                       // context for non-input saved data
} Function;

// memory arena for allocations
typedef struct Arena {
    void *memory;
    size_t capacity;
    size_t offset;
} Arena;

Function *arena_alloc_function(void);
void arena_free(void);

// public api
void backward(Tensor *loss);
void accumulate_grad(Tensor *tensor, Tensor *new_grad);
