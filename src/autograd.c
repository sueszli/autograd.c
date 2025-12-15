#include "autograd.h"
#include "tensor.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_RECURSION_DEPTH 1024
#define MAX_ARRAY_SIZE 1000000
#define INITIAL_ARRAY_CAPACITY 16
#define ARRAY_GROWTH_FACTOR 2

//
// Helpers
//

/*
 * Accumulates the gradient into the tensor's .grad field.
 * Handles the first gradient arrival by cloning it.
 * Subsequent calls accumulate via addition.
 *
 * tensor_mut: The tensor to update (mutable).
 * grad:       The gradient to accumulate.
 */
void accumulate_grad(Tensor *tensor_mut, const Tensor *grad) {
    assert(tensor_mut != NULL);

    if (!tensor_mut->requires_grad) {
        return;
    }

    assert(grad != NULL);
    assert(grad->data != NULL || grad->size == 0);

    if (tensor_mut->grad == NULL) {
        // first gradient: clone it to ensure we own the memory and handle broadcasting if necessary
        // tensor_add handles broadcasting if grad shape differs from tensor_mut shape
        Tensor *zeros = tensor_zeros(tensor_mut->shape, tensor_mut->ndim, false);
        tensor_mut->grad = tensor_add(zeros, grad);
        tensor_free(zeros);
        assert(tensor_mut->grad != NULL && "tensor_add failed");
    } else {
        // subsequent gradients: accumulate via addition
        Tensor *new_grad = tensor_add(tensor_mut->grad, grad);
        assert(new_grad != NULL && "tensor_add failed");
        tensor_free(tensor_mut->grad);
        tensor_mut->grad = new_grad;
    }
}

//
// Topological Sort
//

/*
 * Dynamic array of GradFn pointers.
 * Used for building the topological sort of the graph.
 */
typedef struct {
    GradFn **data;
    uint64_t count;
    uint64_t capacity;
} PtrArray;

static void ptr_array_init(PtrArray *array) {
    assert(array != NULL);
    array->count = 0;
    array->capacity = INITIAL_ARRAY_CAPACITY;
    array->data = (GradFn **)malloc(array->capacity * sizeof(GradFn *));
    assert(array->data != NULL && "malloc failed");
}

static void ptr_array_free(PtrArray *array) {
    assert(array != NULL);
    if (array->data) {
        free(array->data);
        array->data = NULL;
    }
    array->count = 0;
    array->capacity = 0;
}

static void ptr_array_append(PtrArray *array, GradFn *grad_fn) {
    assert(array != NULL);
    assert(grad_fn != NULL);

    if (array->count >= array->capacity) {
        // check for overflow to prevent security vulnerabilities
        if (array->capacity > UINT64_MAX / ARRAY_GROWTH_FACTOR) {
            assert(false && "PtrArray capacity overflow");
        }
        uint64_t new_capacity = array->capacity * ARRAY_GROWTH_FACTOR;
        // enforce upper bound on memory usage
        assert(new_capacity <= MAX_ARRAY_SIZE && "Array capacity exceeds maximum limit");

        GradFn **new_data = (GradFn **)realloc(array->data, new_capacity * sizeof(GradFn *));
        assert(new_data != NULL && "realloc failed");
        array->data = new_data;
        array->capacity = new_capacity;
    }
    array->data[array->count++] = grad_fn;
}

static bool ptr_array_contains(const PtrArray *array, const GradFn *grad_fn) {
    assert(array != NULL);
    // Linear search is O(N), acceptable for expected graph sizes (bounded by MAX_ARRAY_SIZE)
    for (uint64_t idx = 0; idx < array->count; idx++) {
        if (array->data[idx] == grad_fn) {
            return true;
        }
    }
    return false;
}

/*
 * Recursive helper for topological sort via DFS.
 * Post-order traversal ensures dependencies come before dependents.
 *
 * grad_fn: current node in the computation graph.
 * topo:    list to append nodes to in topological order.
 * visited: set of visited nodes to avoid cycles/revisiting.
 * depth:   current recursion depth for stack overflow protection.
 */
static void build_topo_recursive(GradFn *grad_fn, PtrArray *topo, PtrArray *visited, uint64_t depth) {
    assert(depth < MAX_RECURSION_DEPTH && "Recursion depth exceeded: graph too deep");
    assert(grad_fn != NULL);
    assert(topo != NULL);
    assert(visited != NULL);

    if (ptr_array_contains(visited, grad_fn)) {
        return;
    }
    ptr_array_append(visited, grad_fn);

    // ensure dependencies are processed before dependents via post-order traversal
    for (int64_t child_idx = 0; child_idx < grad_fn->next_fn_count; child_idx++) {
        if (grad_fn->next_fns[child_idx]) {
            build_topo_recursive(grad_fn->next_fns[child_idx], topo, visited, depth + 1);
        }
    }

    // add current node after all children have been visited
    ptr_array_append(topo, grad_fn);
}

//
// Autograd Engine
//

/*
 * Computes gradients of root tensor w.r.t. all graph leaves via backpropagation.
 * Uses reverse-mode automatic differentiation: builds computation graph topology,
 * then propagates gradients from root to leaves in reverse topological order.
 */
void backward(Tensor *root, const Tensor *grad) {
    assert(root != NULL);

    // seed the gradient at root node
    if (root->grad == NULL) {
        if (grad == NULL) {
            // scalar loss with implicit gradient of 1.0
            if (root->size == 1) {
                const uint64_t shape[] = {1};
                root->grad = tensor_create(NULL, shape, 0, false);
                root->grad->data[0] = 1.0f;
            } else {
                assert(false && "Grad must be specified for non-scalar root");
            }
        } else {
            // explicit gradient provided by caller
            assert(grad->data != NULL || grad->size == 0);
            // copy gradient to ensure we own the memory
            root->grad = tensor_create(grad->data, grad->shape, grad->ndim, false);
        }
    } else {
        // gradient already exists, accumulate new gradient
        if (grad != NULL) {
            accumulate_grad(root, grad);
        }
    }

    if (!root->grad_fn) {
        // leaf node has no backward function to propagate through
        return;
    }

    // build topological sort of computation graph
    PtrArray topo;
    ptr_array_init(&topo);
    PtrArray visited;
    ptr_array_init(&visited);

    build_topo_recursive(root->grad_fn, &topo, &visited, 0);

    // propagate gradients in reverse topological order (outputs to inputs)
    for (int64_t node_index = (int64_t)topo.count - 1; node_index >= 0; node_index--) {
        GradFn *grad_fn = topo.data[node_index];
        assert(grad_fn != NULL);

        Tensor *output_tensor = grad_fn->out_tensor;
        // propagate gradient only if output has accumulated gradient
        if (output_tensor && output_tensor->grad) {
            grad_fn->apply(grad_fn, output_tensor->grad);
        }
    }

    ptr_array_free(&topo);
    ptr_array_free(&visited);
}

void grad_fn_init(GradFn *fn, void (*apply)(GradFn *, const struct Tensor *), void (*destroy)(GradFn *), GradFn **next_fns, int64_t next_fn_count, const char *name) {
    assert(fn != NULL);
    assert(apply != NULL);
    // next_fns can be NULL if next_fn_count is 0

    fn->apply = apply;
    fn->destroy = destroy;
    fn->next_fns = next_fns;
    fn->next_fn_count = next_fn_count;
    fn->name = name ? strdup(name) : NULL;
    fn->out_tensor = NULL;
}

void grad_fn_free(GradFn *fn) {
    if (!fn) {
        return;
    }
    // Subclass specific cleanup
    if (fn->destroy) {
        fn->destroy(fn);
    }
    // Base cleanup
    if (fn->next_fns) {
        free(fn->next_fns);
    }
    if (fn->name) {
        free(fn->name);
    }
    free(fn);
}
