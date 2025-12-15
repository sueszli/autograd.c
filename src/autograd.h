#pragma once

#include "tensor.h"
#include <stdbool.h>

// Forward declaration
struct Tensor;

// A generic function in the autograd graph.
// specific operations (AddBackward, MulBackward) will "inherit" this
// by embedding it or being castable to it, or we use a union/void* context.
// For Simplicity in C, we'll use a context pointer and a function pointer.
typedef struct GradFn {
    // The apply function: computes gradients and accumulates them into input tensors.
    void (*apply)(struct GradFn *self, struct Tensor *grad_output);

    // References to the parents in the graph (the GradFns of the inputs to this operation).
    // We do NOT own these pointers (they are owned by the Tensors or by the graph structure implied by Tensors).
    struct GradFn **next_fns;
    int num_next;

    // Operation specific data (saved tensors, etc.)
    // We can use a flexible array member or void* context.
    // Let's use specific structs that "inherit" GradFn if we can, or just void* context.
    // To keep it simple and uniform, let's stick to "subclassing" via pointer casing
    // or just embedding context.
    // Given C, "subclassing" by having GradFn as the first member of specific structs is standard.
    // For simplicity in this implementation, we keep a reference to the output tensor
    // to retrieve the incoming gradient during backward pass.
    struct Tensor *out_tensor;
    
    char *name; // For debugging
} GradFn;

// Core engine
void backward(struct Tensor *root, struct Tensor *grad);

// Helper to create a generic GradFn (usually used by specific ops to init their base)
void grad_fn_init(GradFn *fn, void (*apply)(GradFn *, struct Tensor *), GradFn **next_fns, int num_next, char *name);

// Freeing a GradFn (usually called when Tensor is freed)
void grad_fn_free(GradFn *fn);

// Op Creators
GradFn *new_add_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_sub_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_mul_backward(struct Tensor *a, struct Tensor *b);
GradFn *new_div_backward(struct Tensor *a, struct Tensor *b);
