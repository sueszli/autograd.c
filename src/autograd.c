#include "autograd.h"
#include "tensor.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_RECURSION_DEPTH 1024
#define MAX_NDIM 32
#define MAX_ARRAY_SIZE 1000000
#define MAX_TENSOR_SIZE 100000000
#define INITIAL_ARRAY_CAPACITY 16
#define ARRAY_GROWTH_FACTOR 2

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef EPSILON
#define EPSILON 1e-7f
#endif

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
static void accumulate_grad(Tensor *tensor_mut, const Tensor *grad) {
    assert(tensor_mut != NULL);

    if (!tensor_mut->requires_grad) {
        return;
    }

    assert(grad != NULL);
    assert(grad->data != NULL || grad->size == 0);

    if (tensor_mut->grad == NULL) {
        // first gradient: clone it to ensure we own the memory
        tensor_mut->grad = tensor_create(grad->data, grad->shape, grad->ndim, false);
        assert(tensor_mut->grad != NULL && "tensor_create failed");
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

//
// Operations Backwards
//

/*
 * Reverses broadcasting that occurred in forward pass.
 * When forward pass broadcast input (e.g., shape (3) -> (2,3)), the backward
 * gradient must be summed along broadcasted dimensions to match input shape.
 *
 * Example:
 *   Forward: A (2,1) + B (3) -> C (2,3)  [both inputs broadcast to (2,3)]
 *   Backward: dL/dC (2,3)
 *     dL/dA = sum(dL/dC, axis=1, keepdim=True) -> (2,1)
 *     dL/dB = sum(dL/dC, axis=0) -> (3)
 */
static Tensor *unbroadcast(const Tensor *grad, const Tensor *input) {
    if (!grad || !input) {
        return NULL;
    }

    const Tensor *current_grad = grad;
    bool owns_current_grad = false;

    // broadcasting adds dimensions on the left, so collapse extra leading dimensions
    // example: grad (2,3,4) with input (3,4) -> sum axis 0 -> (3,4)
    while (current_grad->ndim > input->ndim) {
        Tensor *summed = tensor_sum(current_grad, 0, false);
        if (owns_current_grad) {
            tensor_free((Tensor *)current_grad);
        }
        current_grad = summed;
        owns_current_grad = true;
    }

    // now dimensions match; collapse any dims where input had size 1
    // example: grad (2,3) with input (2,1) -> sum axis 1 with keepdim -> (2,1)
    assert(current_grad->ndim == input->ndim);
    assert(input->ndim < MAX_NDIM && "Number of dimensions exceeds maximum");

    for (uint64_t dim_idx = 0; dim_idx < input->ndim; dim_idx++) {
        // dimension was broadcasted from 1 to N, so sum back to 1
        if (input->shape[dim_idx] == 1 && current_grad->shape[dim_idx] > 1) {
            Tensor *summed = tensor_sum(current_grad, (int64_t)dim_idx, true);
            if (owns_current_grad) {
                tensor_free((Tensor *)current_grad);
            }
            current_grad = summed;
            owns_current_grad = true;
        }
    }

    // caller expects to own result, so clone if we never created a new tensor
    if (!owns_current_grad) {
        return tensor_create(grad->data, grad->shape, grad->ndim, false);
    }

    return (Tensor *)current_grad;
}

static void accumulate_grad_unbroadcast(Tensor *t, const Tensor *grad) {
    if (!t || !grad || !t->requires_grad) {
        return;
    }

    Tensor *adj_grad = unbroadcast(grad, t);
    if (adj_grad) {
        accumulate_grad(t, adj_grad);
        tensor_free(adj_grad);
    }
}

//
// Specific Backward Implementations
//

// --- Add ---
typedef struct {
    GradFn base;
    Tensor *a;
    Tensor *b;
} AddBackward;

static void add_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    AddBackward *self = (AddBackward *)base;

    accumulate_grad_unbroadcast(self->a, grad_output);
    accumulate_grad_unbroadcast(self->b, grad_output);
}

GradFn *new_add_backward(Tensor *a, Tensor *b) {
    assert(a != NULL);
    assert(b != NULL);

    AddBackward *fn = (AddBackward *)malloc(sizeof(AddBackward));
    assert(fn != NULL && "malloc failed");

    GradFn **next_fns = (GradFn **)malloc(2 * sizeof(GradFn *));
    assert(next_fns != NULL && "malloc failed");

    int64_t next_fn_count = 0;
    if (a->grad_fn)
        next_fns[next_fn_count++] = a->grad_fn;
    if (b->grad_fn)
        next_fns[next_fn_count++] = b->grad_fn;

    grad_fn_init((GradFn *)fn, add_apply, NULL, next_fns, next_fn_count, "AddBackward");
    fn->a = a;
    fn->b = b;
    return (GradFn *)fn;
}

// --- Sub ---
typedef struct {
    GradFn base;
    Tensor *a;
    Tensor *b;
} SubBackward;

static void sub_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    SubBackward *self = (SubBackward *)base;

    // dA = dOut
    accumulate_grad_unbroadcast(self->a, grad_output);

    // dB = -dOut
    // Create zero tensor to subtract from
    Tensor *zeros = tensor_zeros(grad_output->shape, grad_output->ndim, false);
    Tensor *neg_grad = tensor_sub(zeros, grad_output);
    tensor_free(zeros);

    accumulate_grad_unbroadcast(self->b, neg_grad);
    tensor_free(neg_grad);
}

GradFn *new_sub_backward(Tensor *a, Tensor *b) {
    assert(a != NULL);
    assert(b != NULL);

    SubBackward *fn = (SubBackward *)malloc(sizeof(SubBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(2 * sizeof(GradFn *));
    assert(next_fns != NULL);

    int64_t next_fn_count = 0;
    if (a->grad_fn)
        next_fns[next_fn_count++] = a->grad_fn;
    if (b->grad_fn)
        next_fns[next_fn_count++] = b->grad_fn;

    grad_fn_init((GradFn *)fn, sub_apply, NULL, next_fns, next_fn_count, "SubBackward");
    fn->a = a;
    fn->b = b;
    return (GradFn *)fn;
}

// --- Mul ---
typedef struct {
    GradFn base;
    Tensor *a;
    Tensor *b;
} MulBackward;

static void mul_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    MulBackward *self = (MulBackward *)base;

    // dA = dOut * B
    Tensor *da = tensor_mul(grad_output, self->b);
    accumulate_grad_unbroadcast(self->a, da);
    tensor_free(da);

    // dB = dOut * A
    Tensor *db = tensor_mul(grad_output, self->a);
    accumulate_grad_unbroadcast(self->b, db);
    tensor_free(db);
}

GradFn *new_mul_backward(Tensor *a, Tensor *b) {
    assert(a != NULL);
    assert(b != NULL);

    MulBackward *fn = (MulBackward *)malloc(sizeof(MulBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(2 * sizeof(GradFn *));
    assert(next_fns != NULL);

    int64_t next_fn_count = 0;
    if (a->grad_fn)
        next_fns[next_fn_count++] = a->grad_fn;
    if (b->grad_fn)
        next_fns[next_fn_count++] = b->grad_fn;

    grad_fn_init((GradFn *)fn, mul_apply, NULL, next_fns, next_fn_count, "MulBackward");
    fn->a = a;
    fn->b = b;
    return (GradFn *)fn;
}

// --- Div ---
typedef struct {
    GradFn base;
    Tensor *a;
    Tensor *b;
} DivBackward;

static void div_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    DivBackward *self = (DivBackward *)base;

    // dA = dOut / B
    Tensor *da = tensor_div(grad_output, self->b);
    accumulate_grad_unbroadcast(self->a, da);
    tensor_free(da);

    // dB = -dOut * A / (B^2)
    //    = -(dOut * A) / (B * B)

    // -dOut
    Tensor *zeros = tensor_zeros(grad_output->shape, grad_output->ndim, false);
    Tensor *neg_grad = tensor_sub(zeros, grad_output);
    tensor_free(zeros);

    Tensor *num = tensor_mul(neg_grad, self->a);
    tensor_free(neg_grad);

    Tensor *b_sq = tensor_mul(self->b, self->b);
    Tensor *db = tensor_div(num, b_sq);
    tensor_free(num);
    tensor_free(b_sq);

    accumulate_grad_unbroadcast(self->b, db);
    tensor_free(db);
}

GradFn *new_div_backward(Tensor *a, Tensor *b) {
    assert(a != NULL);
    assert(b != NULL);

    DivBackward *fn = (DivBackward *)malloc(sizeof(DivBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(2 * sizeof(GradFn *));
    assert(next_fns != NULL);

    int64_t next_fn_count = 0;
    if (a->grad_fn)
        next_fns[next_fn_count++] = a->grad_fn;
    if (b->grad_fn)
        next_fns[next_fn_count++] = b->grad_fn;

    grad_fn_init((GradFn *)fn, div_apply, NULL, next_fns, next_fn_count, "DivBackward");
    fn->a = a;
    fn->b = b;
    return (GradFn *)fn;
}

// --- Matmul ---
typedef struct {
    GradFn base;
    Tensor *a;
    Tensor *b;
} MatmulBackward;

static void matmul_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    MatmulBackward *self = (MatmulBackward *)base;

    // C = A @ B
    // dA = dC @ B.T
    // dB = A.T @ dC

    // dA
    Tensor *b_T = tensor_transpose(self->b, 0, 1);
    Tensor *da = tensor_matmul(grad_output, b_T);
    accumulate_grad(self->a, da);
    tensor_free(b_T);
    tensor_free(da);

    // dB
    Tensor *a_T = tensor_transpose(self->a, 0, 1);
    Tensor *db = tensor_matmul(a_T, grad_output);
    accumulate_grad(self->b, db);
    tensor_free(a_T);
    tensor_free(db);
}

GradFn *new_matmul_backward(Tensor *a, Tensor *b) {
    assert(a != NULL);
    assert(b != NULL);

    MatmulBackward *fn = (MatmulBackward *)malloc(sizeof(MatmulBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(2 * sizeof(GradFn *));
    assert(next_fns != NULL);

    int64_t next_fn_count = 0;
    if (a->grad_fn)
        next_fns[next_fn_count++] = a->grad_fn;
    if (b->grad_fn)
        next_fns[next_fn_count++] = b->grad_fn;

    grad_fn_init((GradFn *)fn, matmul_apply, NULL, next_fns, next_fn_count, "MatmulBackward");
    fn->a = a;
    fn->b = b;
    return (GradFn *)fn;
}

// --- Sum ---
typedef struct {
    GradFn base;
    Tensor *input;
    int64_t dim_idx;
    bool keepdims;
} SumBackward;

static void sum_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    SumBackward *self = (SumBackward *)base;

    // If keepdims=true, grad_output shape matches required broadcast input.
    // If keepdims=false, we need to inject the dimension with size 1 back in.

    const Tensor *grad_expanded = grad_output;
    bool needs_free = false;

    if (!self->keepdims) {
        int64_t ndim = (int64_t)self->input->ndim;
        int64_t new_shape[MAX_NDIM] = {0};

        int64_t target_dim = (self->dim_idx < 0) ? (self->dim_idx + ndim) : self->dim_idx;
        assert(target_dim >= 0 && target_dim < ndim && "target_dim out of bounds");

        int64_t grad_dim_idx = 0;
        for (int64_t dim_idx = 0; dim_idx < ndim; dim_idx++) {
            if (dim_idx == target_dim) {
                new_shape[dim_idx] = 1;
            } else {
                new_shape[dim_idx] = (int64_t)grad_output->shape[grad_dim_idx++];
            }
        }

        grad_expanded = tensor_reshape(grad_output, new_shape, (uint64_t)ndim);
        needs_free = true;
    }

    // Now grad_expanded has the shape of input (with 1s in summed dims).
    // Broadcasting handles the expansion.
    accumulate_grad(self->input, grad_expanded);

    if (needs_free) {
        tensor_free((Tensor *)grad_expanded);
    }
}

GradFn *new_sum_backward(Tensor *input, int64_t dim_idx, bool keepdims) {
    assert(input != NULL);

    SumBackward *fn = (SumBackward *)malloc(sizeof(SumBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);

    int64_t next_fn_count = 0;
    if (input->grad_fn)
        next_fns[next_fn_count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, sum_apply, NULL, next_fns, next_fn_count, "SumBackward");
    fn->input = input;
    fn->dim_idx = dim_idx;
    fn->keepdims = keepdims;
    return (GradFn *)fn;
}

// --- ReLU ---
typedef struct {
    GradFn base;
    Tensor *input;
} ReluBackward;

static void relu_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    ReluBackward *self = (ReluBackward *)base;

    assert(grad_output->size == self->input->size && "ReLU grad shape mismatch");
    assert(grad_output->size < MAX_TENSOR_SIZE && "Tensor size exceeds maximum limit");
    Tensor *grad_input = tensor_zeros(grad_output->shape, grad_output->ndim, false);

    for (uint64_t elem_idx = 0; elem_idx < grad_output->size; elem_idx++) {
        if (self->input->data[elem_idx] > 0.0f) {
            grad_input->data[elem_idx] = grad_output->data[elem_idx];
        } else {
            grad_input->data[elem_idx] = 0.0f;
        }
    }

    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

GradFn *new_relu_backward(Tensor *input) {
    assert(input != NULL);

    ReluBackward *fn = (ReluBackward *)malloc(sizeof(ReluBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);

    int64_t next_fn_count = 0;
    if (input->grad_fn)
        next_fns[next_fn_count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, relu_apply, NULL, next_fns, next_fn_count, "ReluBackward");
    fn->input = input;
    return (GradFn *)fn;
}

// --- Sigmoid ---
typedef struct {
    GradFn base;
    Tensor *input;
    Tensor *output;
} SigmoidBackward;

static void sigmoid_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    SigmoidBackward *self = (SigmoidBackward *)base;

    // grad_input = grad_output * sigmoid(x) * (1 - sigmoid(x))
    //            = grad_output * output * (1 - output)

    // 1 (tensor)
    Tensor *ones = tensor_zeros(self->output->shape, self->output->ndim, false);
    for (uint64_t elem_idx = 0; elem_idx < ones->size; elem_idx++)
        ones->data[elem_idx] = 1.0f;

    // (1 - output)
    Tensor *one_minus_out = tensor_sub(ones, self->output);
    tensor_free(ones);

    // output * (1 - output)
    Tensor *d_sigmoid = tensor_mul(self->output, one_minus_out);
    tensor_free(one_minus_out);

    // grad_output * ...
    Tensor *grad_input = tensor_mul(grad_output, d_sigmoid);
    tensor_free(d_sigmoid);

    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

GradFn *new_sigmoid_backward(Tensor *input, Tensor *output) {
    assert(input != NULL);
    assert(output != NULL);

    SigmoidBackward *fn = (SigmoidBackward *)malloc(sizeof(SigmoidBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);

    int64_t next_fn_count = 0;
    if (input->grad_fn)
        next_fns[next_fn_count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, sigmoid_apply, NULL, next_fns, next_fn_count, "SigmoidBackward");
    fn->input = input;
    fn->output = output;
    return (GradFn *)fn;
}

// --- Softmax ---
typedef struct {
    GradFn base;
    Tensor *input;
    Tensor *output;
    int64_t dim;
} SoftmaxBackward;

static void softmax_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    SoftmaxBackward *self = (SoftmaxBackward *)base;

    // dL/dx_i = y_i * (dL/dy_i - sum_j(y_j * dL/dy_j))
    // Term inside parens: grad_output - sum(output * grad_output)

    Tensor *prod = tensor_mul(grad_output, self->output);
    Tensor *sum_prod = tensor_sum(prod, self->dim, true);
    tensor_free(prod);

    Tensor *sub = tensor_sub(grad_output, sum_prod);
    tensor_free(sum_prod);

    Tensor *grad_input = tensor_mul(self->output, sub);
    tensor_free(sub);

    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

GradFn *new_softmax_backward(Tensor *input, Tensor *output, int64_t dim) {
    assert(input != NULL);
    assert(output != NULL);

    SoftmaxBackward *fn = (SoftmaxBackward *)malloc(sizeof(SoftmaxBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);

    int64_t next_fn_count = 0;
    if (input->grad_fn)
        next_fns[next_fn_count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, softmax_apply, NULL, next_fns, next_fn_count, "SoftmaxBackward");
    fn->input = input;
    fn->output = output;
    fn->dim = dim;
    return (GradFn *)fn;
}

// --- Reshape ---
typedef struct {
    GradFn base;
    Tensor *input;
    uint64_t *old_shape;
    uint64_t old_ndim;
} ReshapeBackward;

static void reshape_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    ReshapeBackward *self = (ReshapeBackward *)base;

    int64_t signed_shape[MAX_NDIM] = {0};
    for (uint64_t dim_idx = 0; dim_idx < self->old_ndim; dim_idx++) {
        signed_shape[dim_idx] = (int64_t)self->old_shape[dim_idx];
    }

    // Reshape back to the original input shape
    Tensor *grad_input = tensor_reshape(grad_output, signed_shape, self->old_ndim);
    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

static void reshape_destroy(GradFn *fn) {
    ReshapeBackward *self = (ReshapeBackward *)fn;
    if (self->old_shape) {
        free(self->old_shape);
    }
}

GradFn *new_reshape_backward(Tensor *input, const uint64_t *old_shape, uint64_t old_ndim) {
    assert(input != NULL);
    assert(old_shape != NULL);

    ReshapeBackward *fn = (ReshapeBackward *)malloc(sizeof(ReshapeBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);

    int64_t next_fn_count = 0;
    if (input->grad_fn)
        next_fns[next_fn_count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, reshape_apply, reshape_destroy, next_fns, next_fn_count, "ReshapeBackward");
    fn->input = input;
    fn->old_ndim = old_ndim;
    fn->old_shape = (uint64_t *)malloc(old_ndim * sizeof(uint64_t));
    assert(fn->old_shape != NULL);
    memcpy(fn->old_shape, old_shape, old_ndim * sizeof(uint64_t));

    return (GradFn *)fn;
}

// --- Transpose ---
typedef struct {
    GradFn base;
    Tensor *input;
    uint64_t dim0;
    uint64_t dim1;
} TransposeBackward;

static void transpose_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    TransposeBackward *self = (TransposeBackward *)base;

    // Grad input = grad_output.transpose(dim0, dim1)
    // Transpose is its own inverse operation in terms of swapping dims
    Tensor *grad_input = tensor_transpose(grad_output, self->dim0, self->dim1);
    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

GradFn *new_transpose_backward(Tensor *input, uint64_t dim0, uint64_t dim1) {
    assert(input != NULL);
    TransposeBackward *fn = (TransposeBackward *)malloc(sizeof(TransposeBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);
    int64_t next_fn_count = 0;
    if (input->grad_fn)
        next_fns[next_fn_count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, transpose_apply, NULL, next_fns, next_fn_count, "TransposeBackward");
    fn->input = input;
    fn->dim0 = dim0;
    fn->dim1 = dim1;
    return (GradFn *)fn;
}

// --- GetItem (Slice) ---
typedef struct {
    GradFn base;
    Tensor *input;
    uint64_t *multidim;
} GetItemBackward;

static void getitem_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    GetItemBackward *self = (GetItemBackward *)base;

    // Create a zero tensor of input shape
    // Assuming we simply fill in the gradient at the specific index.
    Tensor *grad_input = tensor_zeros(self->input->shape, self->input->ndim, false);

    // We need to write grad_output to the correct position.
    // NOTE: tensor_getitem usually returns a scalar (if fully indexed) or a slice?
    // Based on implementation, it seems to handle single element access via multidim array?
    // If grad_output is scalar:
    assert(grad_output->size == 1 && "GetItem backward currently supports scalar output");

    // Calculate linear offset in input tensor
    // TODO: This duplication of logic (multidim_to_linear) is not ideal but acceptable for now.
    // Ideally we should expose multidim_to_linear from tensor.h/c
    // But it's static there. We can reimplement simple logic or assume strides are available.

    if (self->input->strides) {
        uint64_t offset = 0;
        for (uint64_t dim_idx = 0; dim_idx < self->input->ndim; dim_idx++) {
            offset += self->multidim[dim_idx] * self->input->strides[dim_idx];
        }
        assert(offset < grad_input->size);
        grad_input->data[offset] = grad_output->data[0];
    } else {
        // Fallback or error? Strides should exist if tensor created properly.
        assert(false && "Input tensor has no strides");
    }

    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

static void getitem_destroy(GradFn *fn) {
    GetItemBackward *self = (GetItemBackward *)fn;
    if (self->multidim) {
        free(self->multidim);
    }
}

GradFn *new_getitem_backward(Tensor *input, const uint64_t *multidim) {
    assert(input != NULL);
    assert(multidim != NULL);
    GetItemBackward *fn = (GetItemBackward *)malloc(sizeof(GetItemBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);
    int64_t next_fn_count = 0;
    if (input->grad_fn)
        next_fns[next_fn_count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, getitem_apply, getitem_destroy, next_fns, next_fn_count, "GetItemBackward");
    fn->input = input;
    fn->multidim = (uint64_t *)malloc(input->ndim * sizeof(uint64_t));
    assert(fn->multidim != NULL);
    memcpy(fn->multidim, multidim, input->ndim * sizeof(uint64_t));

    return (GradFn *)fn;
}

// --- GELU ---
typedef struct {
    GradFn base;
    Tensor *input;
} GELUBackward;

static void gelu_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    GELUBackward *self = (GELUBackward *)base;

    Tensor *grad_input = tensor_create(NULL, grad_output->shape, grad_output->ndim, false);
    assert(self->input->size < MAX_TENSOR_SIZE && "Tensor size exceeds maximum limit");

    // GELU gradient approximation
    // gelu(x) approx 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float32_t sqrt_2_over_pi = sqrtf(2.0f / (float32_t)M_PI);
    float32_t coeff = 0.044715f;

    for (uint64_t elem_idx = 0; elem_idx < self->input->size; elem_idx++) {
        float32_t x = self->input->data[elem_idx];
        float32_t x2 = x * x;
        float32_t x3 = x2 * x;

        float32_t tanh_arg = sqrt_2_over_pi * (x + coeff * x3);
        float32_t tanh_out = tanhf(tanh_arg);
        float32_t sech_sq = 1.0f - tanh_out * tanh_out;

        float32_t d_tanh_arg = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x2);

        // y = 0.5 * x * (1 + tanh(...))
        // dy/dx = 0.5 * (1 + tanh(...)) + 0.5 * x * sech^2(...) * d(...)/dx
        float32_t gelu_grad = 0.5f * (1.0f + tanh_out) + 0.5f * x * sech_sq * d_tanh_arg;

        grad_input->data[elem_idx] = grad_output->data[elem_idx] * gelu_grad;
    }

    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

GradFn *new_gelu_backward(Tensor *input) {
    assert(input != NULL);
    GELUBackward *fn = (GELUBackward *)malloc(sizeof(GELUBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);
    int64_t next_fn_count = 0;
    if (input->grad_fn)
        next_fns[next_fn_count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, gelu_apply, NULL, next_fns, next_fn_count, "GELUBackward");
    fn->input = input;
    return (GradFn *)fn;
}

// --- MSE ---
typedef struct {
    GradFn base;
    Tensor *predictions;
    Tensor *targets;
} MSEBackward;

static void mse_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    MSEBackward *self = (MSEBackward *)base;

    // loss is scalar, so grad_output is scalar tensor
    assert(grad_output->size == 1);
    float32_t grad_scalar = grad_output->data[0];

    Tensor *grad_pred = tensor_create(NULL, self->predictions->shape, self->predictions->ndim, false);
    assert(self->predictions->size < MAX_TENSOR_SIZE && "Tensor size exceeds maximum limit");
    float32_t element_count_f32 = (float32_t)self->predictions->size;

    for (uint64_t elem_idx = 0; elem_idx < self->predictions->size; elem_idx++) {
        // gradient of MSE: 2 * (pred - target) / N
        float32_t diff = self->predictions->data[elem_idx] - self->targets->data[elem_idx];
        grad_pred->data[elem_idx] = grad_scalar * 2.0f * diff / element_count_f32;
    }

    accumulate_grad(self->predictions, grad_pred);
    tensor_free(grad_pred);
}

GradFn *new_mse_backward(Tensor *predictions, Tensor *targets) {
    assert(predictions != NULL);
    assert(targets != NULL);
    MSEBackward *fn = (MSEBackward *)malloc(sizeof(MSEBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);
    int64_t next_fn_count = 0;
    if (predictions->grad_fn)
        next_fns[next_fn_count++] = predictions->grad_fn;

    grad_fn_init((GradFn *)fn, mse_apply, NULL, next_fns, next_fn_count, "MSEBackward");
    fn->predictions = predictions;
    fn->targets = targets;
    return (GradFn *)fn;
}

// --- BCE ---
typedef struct {
    GradFn base;
    Tensor *predictions;
    Tensor *targets;
} BCEBackward;

static void bce_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    BCEBackward *self = (BCEBackward *)base;

    assert(grad_output->size == 1);
    float32_t grad_scalar = grad_output->data[0];

    Tensor *grad_pred = tensor_create(NULL, self->predictions->shape, self->predictions->ndim, false);
    assert(self->predictions->size < MAX_TENSOR_SIZE && "Tensor size exceeds maximum limit");
    float32_t element_count_f32 = (float32_t)self->predictions->size;

    for (uint64_t elem_idx = 0; elem_idx < self->predictions->size; elem_idx++) {
        float32_t pred = self->predictions->data[elem_idx];
        float32_t target = self->targets->data[elem_idx];

        // clamp prediction to avoid log(0) in forward pass
        if (pred < EPSILON)
            pred = EPSILON;
        if (pred > 1.0f - EPSILON)
            pred = 1.0f - EPSILON;

        // gradient of BCE: (pred - target) / (pred * (1 - pred)) / N
        // derived from: -mean(target*log(pred) + (1-target)*log(1-pred))
        float32_t denom = pred * (1.0f - pred) * element_count_f32;
        grad_pred->data[elem_idx] = grad_scalar * (pred - target) / denom;
    }

    accumulate_grad(self->predictions, grad_pred);
    tensor_free(grad_pred);
}

GradFn *new_bce_backward(Tensor *predictions, Tensor *targets) {
    assert(predictions != NULL);
    assert(targets != NULL);
    BCEBackward *fn = (BCEBackward *)malloc(sizeof(BCEBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);
    int64_t next_fn_count = 0;
    if (predictions->grad_fn)
        next_fns[next_fn_count++] = predictions->grad_fn;

    grad_fn_init((GradFn *)fn, bce_apply, NULL, next_fns, next_fn_count, "BCEBackward");
    fn->predictions = predictions;
    fn->targets = targets;
    return (GradFn *)fn;
}

// --- Tanh ---
typedef struct {
    GradFn base;
    Tensor *input;
    Tensor *output;
} TanhBackward;

static void tanh_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    TanhBackward *self = (TanhBackward *)base;

    // grad_input = grad_output * (1 - tanh(x)^2)
    //            = grad_output * (1 - output^2)

    // output^2
    Tensor *output_sq = tensor_mul(self->output, self->output);

    // 1 (tensor)
    Tensor *ones = tensor_zeros(self->output->shape, self->output->ndim, false);
    for (uint64_t elem_idx = 0; elem_idx < ones->size; elem_idx++)
        ones->data[elem_idx] = 1.0f;

    // (1 - output^2)
    Tensor *one_minus_out_sq = tensor_sub(ones, output_sq);
    tensor_free(ones);
    tensor_free(output_sq);

    // grad_output * (1 - output^2)
    Tensor *grad_input = tensor_mul(grad_output, one_minus_out_sq);
    tensor_free(one_minus_out_sq);

    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

GradFn *new_tanh_backward(Tensor *input, Tensor *output) {
    assert(input != NULL);
    assert(output != NULL);

    TanhBackward *fn = (TanhBackward *)malloc(sizeof(TanhBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);

    int64_t next_fn_count = 0;
    if (input->grad_fn)
        next_fns[next_fn_count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, tanh_apply, NULL, next_fns, next_fn_count, "TanhBackward");
    fn->input = input;
    fn->output = output;
    return (GradFn *)fn;
}

// --- Mean ---
typedef struct {
    GradFn base;
    Tensor *input;
    int64_t dim_idx;
    bool keepdims;
} MeanBackward;

static void mean_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    MeanBackward *self = (MeanBackward *)base;

    // Mean is sum / N, so gradient is distributed: grad_input = grad_output / N
    // Similar to sum backward but divide by dimension size

    const Tensor *grad_expanded = grad_output;
    bool needs_free = false;

    if (!self->keepdims) {
        int64_t ndim = (int64_t)self->input->ndim;
        int64_t new_shape[MAX_NDIM] = {0};

        int64_t target_dim = (self->dim_idx < 0) ? (self->dim_idx + ndim) : self->dim_idx;
        assert(target_dim >= 0 && target_dim < ndim && "target_dim out of bounds");

        int64_t grad_dim_idx = 0;
        for (int64_t dim_idx = 0; dim_idx < ndim; dim_idx++) {
            if (dim_idx == target_dim) {
                new_shape[dim_idx] = 1;
            } else {
                new_shape[dim_idx] = (int64_t)grad_output->shape[grad_dim_idx++];
            }
        }

        grad_expanded = tensor_reshape(grad_output, new_shape, (uint64_t)ndim);
        needs_free = true;
    }

    // Get the size of the dimension we averaged over
    int64_t ndim = (int64_t)self->input->ndim;
    int64_t target_dim = (self->dim_idx < 0) ? (self->dim_idx + ndim) : self->dim_idx;
    assert(target_dim >= 0 && target_dim < ndim);
    uint64_t dim_size = self->input->shape[target_dim];

    // Divide gradient by dimension size
    Tensor *dim_size_tensor = tensor_zeros(grad_expanded->shape, grad_expanded->ndim, false);
    for (uint64_t elem_idx = 0; elem_idx < dim_size_tensor->size; elem_idx++)
        dim_size_tensor->data[elem_idx] = (float32_t)dim_size;

    Tensor *grad_scaled = tensor_div(grad_expanded, dim_size_tensor);
    tensor_free(dim_size_tensor);

    // Now grad_scaled has the shape of input (with 1s in averaged dims).
    // Broadcasting handles the expansion.
    accumulate_grad(self->input, grad_scaled);

    tensor_free(grad_scaled);
    if (needs_free) {
        tensor_free((Tensor *)grad_expanded);
    }
}

GradFn *new_mean_backward(Tensor *input, int64_t dim_idx, bool keepdims) {
    assert(input != NULL);

    MeanBackward *fn = (MeanBackward *)malloc(sizeof(MeanBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);

    int64_t next_fn_count = 0;
    if (input->grad_fn)
        next_fns[next_fn_count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, mean_apply, NULL, next_fns, next_fn_count, "MeanBackward");
    fn->input = input;
    fn->dim_idx = dim_idx;
    fn->keepdims = keepdims;
    return (GradFn *)fn;
}

// --- Max ---
typedef struct {
    GradFn base;
    Tensor *input;
    Tensor *output;
    int64_t dim_idx;
    bool keepdims;
} MaxBackward;

static void max_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    MaxBackward *self = (MaxBackward *)base;

    // Gradient flows only to the maximum elements
    // For each slice along dim_idx, find which elements == max, distribute gradient to them

    const Tensor *grad_expanded = grad_output;
    bool needs_free_expanded = false;

    if (!self->keepdims) {
        int64_t ndim = (int64_t)self->input->ndim;
        int64_t new_shape[MAX_NDIM] = {0};

        int64_t target_dim = (self->dim_idx < 0) ? (self->dim_idx + ndim) : self->dim_idx;
        assert(target_dim >= 0 && target_dim < ndim && "target_dim out of bounds");

        int64_t grad_dim_idx = 0;
        for (int64_t dim_idx = 0; dim_idx < ndim; dim_idx++) {
            if (dim_idx == target_dim) {
                new_shape[dim_idx] = 1;
            } else {
                new_shape[dim_idx] = (int64_t)grad_output->shape[grad_dim_idx++];
            }
        }

        grad_expanded = tensor_reshape(grad_output, new_shape, (uint64_t)ndim);
        needs_free_expanded = true;
    }

    // Expand output to match input shape (if keepdims was false, it's now expanded via reshape)
    const Tensor *output_expanded = self->output;
    bool needs_free_output = false;

    if (!self->keepdims) {
        int64_t ndim = (int64_t)self->input->ndim;
        int64_t new_shape[MAX_NDIM] = {0};

        int64_t target_dim = (self->dim_idx < 0) ? (self->dim_idx + ndim) : self->dim_idx;

        int64_t output_dim_idx = 0;
        for (int64_t dim_idx = 0; dim_idx < ndim; dim_idx++) {
            if (dim_idx == target_dim) {
                new_shape[dim_idx] = 1;
            } else {
                new_shape[dim_idx] = (int64_t)self->output->shape[output_dim_idx++];
            }
        }

        output_expanded = tensor_reshape(self->output, new_shape, (uint64_t)ndim);
        needs_free_output = true;
    }

    // Create gradient for input
    Tensor *grad_input = tensor_zeros(self->input->shape, self->input->ndim, false);

    // For each element, if it equals the max value, pass through the gradient
    // This requires broadcasting logic
    assert(self->input->size < MAX_TENSOR_SIZE && "Tensor size exceeds maximum limit");

    for (uint64_t elem_idx = 0; elem_idx < self->input->size; elem_idx++) {
        // Get corresponding index in output_expanded and grad_expanded (with broadcasting)
        // Calculate the broadcasted index
        uint64_t remaining = elem_idx;
        uint64_t broadcast_idx = 0;
        uint64_t broadcast_stride = 1;

        for (int64_t dim = (int64_t)self->input->ndim - 1; dim >= 0; dim--) {
            uint64_t idx_in_dim = remaining % self->input->shape[dim];
            remaining /= self->input->shape[dim];

            // For output, if dimension is 1 (broadcasted), use index 0
            uint64_t out_idx_in_dim = (output_expanded->shape[dim] == 1) ? 0 : idx_in_dim;
            broadcast_idx += out_idx_in_dim * broadcast_stride;
            broadcast_stride *= output_expanded->shape[dim];
        }

        // Check if this input element is the maximum
        if (fabsf(self->input->data[elem_idx] - output_expanded->data[broadcast_idx]) < EPSILON) {
            grad_input->data[elem_idx] = grad_expanded->data[broadcast_idx];
        }
    }

    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);

    if (needs_free_expanded) {
        tensor_free((Tensor *)grad_expanded);
    }
    if (needs_free_output) {
        tensor_free((Tensor *)output_expanded);
    }
}

GradFn *new_max_backward(Tensor *input, Tensor *output, int64_t dim_idx, bool keepdims) {
    assert(input != NULL);
    assert(output != NULL);

    MaxBackward *fn = (MaxBackward *)malloc(sizeof(MaxBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);

    int64_t next_fn_count = 0;
    if (input->grad_fn)
        next_fns[next_fn_count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, max_apply, NULL, next_fns, next_fn_count, "MaxBackward");
    fn->input = input;
    fn->output = output;
    fn->dim_idx = dim_idx;
    fn->keepdims = keepdims;
    return (GradFn *)fn;
}

// --- CrossEntropy ---
typedef struct {
    GradFn base;
    Tensor *logits;
    Tensor *targets;
} CrossEntropyBackward;

/*
 * Helper to compute softmax probability for a single element.
 * Uses numerically stable softmax with max subtraction.
 */
static inline float32_t compute_softmax_prob(const float32_t *logits, uint64_t offset, float32_t max_val, float32_t sum_exp, uint64_t class_idx) { return expf(logits[offset + class_idx] - max_val) / sum_exp; }

static void crossentropy_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    CrossEntropyBackward *self = (CrossEntropyBackward *)base;

    assert(grad_output->size == 1);
    float32_t grad_scalar = grad_output->data[0];

    // cross-entropy operates on 2D logits: (batch_size, class_count)
    assert(self->logits->ndim == 2);
    uint64_t batch_size = self->logits->shape[0];
    uint64_t class_count = self->logits->shape[1];
    assert(batch_size < MAX_TENSOR_SIZE && "Batch size exceeds maximum limit");
    assert(class_count < MAX_TENSOR_SIZE && "Class count exceeds maximum limit");

    Tensor *grad_logits = tensor_create(NULL, self->logits->shape, self->logits->ndim, false);
    float32_t batch_size_f32 = (float32_t)batch_size;

    for (uint64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        uint64_t offset = batch_idx * class_count;

        // find max logit for numerical stability in softmax
        float32_t max_logit = -INFINITY;
        for (uint64_t class_idx = 0; class_idx < class_count; class_idx++) {
            float32_t logit_val = self->logits->data[offset + class_idx];
            if (logit_val > max_logit)
                max_logit = logit_val;
        }

        // compute sum of exponentials for softmax denominator
        float32_t exp_sum = 0.0f;
        for (uint64_t class_idx = 0; class_idx < class_count; class_idx++) {
            exp_sum += expf(self->logits->data[offset + class_idx] - max_logit);
        }

        // gradient: (softmax_prob - one_hot_target) / batch_size
        uint64_t target_class_idx = (uint64_t)self->targets->data[batch_idx];

        for (uint64_t class_idx = 0; class_idx < class_count; class_idx++) {
            float32_t prob = compute_softmax_prob(self->logits->data, offset, max_logit, exp_sum, class_idx);
            float32_t one_hot_target = (class_idx == target_class_idx) ? 1.0f : 0.0f;

            grad_logits->data[offset + class_idx] = grad_scalar * (prob - one_hot_target) / batch_size_f32;
        }
    }

    accumulate_grad(self->logits, grad_logits);
    tensor_free(grad_logits);
}

GradFn *new_crossentropy_backward(Tensor *logits, Tensor *targets) {
    assert(logits != NULL);
    assert(targets != NULL);
    CrossEntropyBackward *fn = (CrossEntropyBackward *)malloc(sizeof(CrossEntropyBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);
    int64_t next_fn_count = 0;
    if (logits->grad_fn)
        next_fns[next_fn_count++] = logits->grad_fn;

    grad_fn_init((GradFn *)fn, crossentropy_apply, NULL, next_fns, next_fn_count, "CrossEntropyBackward");
    fn->logits = logits;
    fn->targets = targets;
    return (GradFn *)fn;
}

// --- Conv2d ---
typedef struct {
    GradFn base;
    Tensor *input;
    Tensor *weight;
    Tensor *bias;
    uint64_t stride;
    uint64_t padding;
    uint64_t kernel_size;
} Conv2dBackward;

static void conv2d_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    Conv2dBackward *self = (Conv2dBackward *)base;

    // Call the existing conv2d_backward function from convolutions.c
    Tensor *grad_input = NULL;
    Tensor *grad_weight = NULL;
    Tensor *grad_bias = NULL;

    // Import the function declaration (we'll need to include convolutions.h)
    extern void conv2d_backward(const Tensor *input, const Tensor *weight, const Tensor *bias, uint64_t stride, uint64_t padding, uint64_t kernel_size, const Tensor *grad_output, Tensor **out_grad_in, Tensor **out_grad_w, Tensor **out_grad_b);

    conv2d_backward(self->input, self->weight, self->bias, self->stride, self->padding, self->kernel_size, grad_output, &grad_input, &grad_weight, &grad_bias);

    // Accumulate gradients
    if (grad_input) {
        accumulate_grad(self->input, grad_input);
        tensor_free(grad_input);
    }

    if (grad_weight) {
        accumulate_grad(self->weight, grad_weight);
        tensor_free(grad_weight);
    }

    if (grad_bias && self->bias) {
        accumulate_grad(self->bias, grad_bias);
        tensor_free(grad_bias);
    }
}

GradFn *new_conv2d_backward(Tensor *input, Tensor *weight, Tensor *bias, uint64_t stride, uint64_t padding, uint64_t kernel_size) {
    assert(input != NULL);
    assert(weight != NULL);

    Conv2dBackward *fn = (Conv2dBackward *)malloc(sizeof(Conv2dBackward));
    assert(fn != NULL);

    // Allocate space for up to 3 next_fns (input, weight, bias)
    GradFn **next_fns = (GradFn **)malloc(3 * sizeof(GradFn *));
    assert(next_fns != NULL);

    int64_t next_fn_count = 0;
    if (input->grad_fn)
        next_fns[next_fn_count++] = input->grad_fn;
    if (weight->grad_fn)
        next_fns[next_fn_count++] = weight->grad_fn;
    if (bias && bias->grad_fn)
        next_fns[next_fn_count++] = bias->grad_fn;

    grad_fn_init((GradFn *)fn, conv2d_apply, NULL, next_fns, next_fn_count, "Conv2dBackward");
    fn->input = input;
    fn->weight = weight;
    fn->bias = bias;
    fn->stride = stride;
    fn->padding = padding;
    fn->kernel_size = kernel_size;
    return (GradFn *)fn;
}

// --- MaxPool2d ---
typedef struct {
    GradFn base;
    Tensor *input;
    uint64_t *output_shape;
    uint64_t kernel_size;
    uint64_t stride;
    uint64_t padding;
} MaxPool2dBackward;

static void maxpool2d_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    MaxPool2dBackward *self = (MaxPool2dBackward *)base;

    // Call the existing maxpool2d_backward function from convolutions.c
    extern Tensor *maxpool2d_backward(const Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding, const Tensor *grad_output);

    Tensor *grad_input = maxpool2d_backward(self->input, self->output_shape, self->kernel_size, self->stride, self->padding, grad_output);

    if (grad_input) {
        accumulate_grad(self->input, grad_input);
        tensor_free(grad_input);
    }
}

static void maxpool2d_destroy(GradFn *fn) {
    MaxPool2dBackward *self = (MaxPool2dBackward *)fn;
    if (self->output_shape) {
        free(self->output_shape);
    }
}

GradFn *new_maxpool2d_backward(Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding) {
    assert(input != NULL);
    assert(output_shape != NULL);

    MaxPool2dBackward *fn = (MaxPool2dBackward *)malloc(sizeof(MaxPool2dBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);

    int64_t next_fn_count = 0;
    if (input->grad_fn)
        next_fns[next_fn_count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, maxpool2d_apply, maxpool2d_destroy, next_fns, next_fn_count, "MaxPool2dBackward");
    fn->input = input;
    fn->kernel_size = kernel_size;
    fn->stride = stride;
    fn->padding = padding;

    // Store a copy of the output_shape (assuming 4D tensor)
    fn->output_shape = (uint64_t *)malloc(4 * sizeof(uint64_t));
    assert(fn->output_shape != NULL);
    memcpy(fn->output_shape, output_shape, 4 * sizeof(uint64_t));

    return (GradFn *)fn;
}

// --- AvgPool2d ---
typedef struct {
    GradFn base;
    Tensor *input;
    uint64_t *output_shape;
    uint64_t kernel_size;
    uint64_t stride;
    uint64_t padding;
} AvgPool2dBackward;

static void avgpool2d_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    AvgPool2dBackward *self = (AvgPool2dBackward *)base;

    // Call the existing avgpool2d_backward function from convolutions.c
    extern Tensor *avgpool2d_backward(const Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding, const Tensor *grad_output);

    Tensor *grad_input = avgpool2d_backward(self->input, self->output_shape, self->kernel_size, self->stride, self->padding, grad_output);

    if (grad_input) {
        accumulate_grad(self->input, grad_input);
        tensor_free(grad_input);
    }
}

static void avgpool2d_destroy(GradFn *fn) {
    AvgPool2dBackward *self = (AvgPool2dBackward *)fn;
    if (self->output_shape) {
        free(self->output_shape);
    }
}

GradFn *new_avgpool2d_backward(Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding) {
    assert(input != NULL);
    assert(output_shape != NULL);

    AvgPool2dBackward *fn = (AvgPool2dBackward *)malloc(sizeof(AvgPool2dBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);

    int64_t next_fn_count = 0;
    if (input->grad_fn)
        next_fns[next_fn_count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, avgpool2d_apply, avgpool2d_destroy, next_fns, next_fn_count, "AvgPool2dBackward");
    fn->input = input;
    fn->kernel_size = kernel_size;
    fn->stride = stride;
    fn->padding = padding;

    // Store a copy of the output_shape (assuming 4D tensor)
    fn->output_shape = (uint64_t *)malloc(4 * sizeof(uint64_t));
    assert(fn->output_shape != NULL);
    memcpy(fn->output_shape, output_shape, 4 * sizeof(uint64_t));

    return (GradFn *)fn;
}

// --- BatchNorm2d ---
typedef struct {
    GradFn base;
    Tensor *input;
    Tensor *gamma;
    Tensor *batch_mean;
    Tensor *batch_var;
    float32_t eps;
} BatchNorm2dBackward;

static void batchnorm2d_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    BatchNorm2dBackward *self = (BatchNorm2dBackward *)base;

    // Call the existing batchnorm2d_backward function from convolutions.c
    Tensor *grad_input = NULL;
    Tensor *grad_gamma = NULL;
    Tensor *grad_beta = NULL;

    extern void batchnorm2d_backward(const Tensor *input, const Tensor *gamma, const Tensor *batch_mean, const Tensor *batch_var, float32_t eps, const Tensor *grad_output, Tensor **out_grad_in, Tensor **out_grad_gamma, Tensor **out_grad_beta);

    batchnorm2d_backward(self->input, self->gamma, self->batch_mean, self->batch_var, self->eps, grad_output, &grad_input, &grad_gamma, &grad_beta);

    // Accumulate gradients
    if (grad_input) {
        accumulate_grad(self->input, grad_input);
        tensor_free(grad_input);
    }

    if (grad_gamma && self->gamma) {
        accumulate_grad(self->gamma, grad_gamma);
        tensor_free(grad_gamma);
    }

    if (grad_beta) {
        // Note: beta gradient would accumulate to a beta parameter if it exists
        // For now we just free it since we don't track beta in the backward context
        tensor_free(grad_beta);
    }
}

GradFn *new_batchnorm2d_backward(Tensor *input, Tensor *gamma, Tensor *batch_mean, Tensor *batch_var, float32_t eps) {
    assert(input != NULL);
    assert(gamma != NULL);
    assert(batch_mean != NULL);
    assert(batch_var != NULL);

    BatchNorm2dBackward *fn = (BatchNorm2dBackward *)malloc(sizeof(BatchNorm2dBackward));
    assert(fn != NULL);

    // Allocate space for up to 2 next_fns (input, gamma)
    GradFn **next_fns = (GradFn **)malloc(2 * sizeof(GradFn *));
    assert(next_fns != NULL);

    int64_t next_fn_count = 0;
    if (input->grad_fn)
        next_fns[next_fn_count++] = input->grad_fn;
    if (gamma->grad_fn)
        next_fns[next_fn_count++] = gamma->grad_fn;

    grad_fn_init((GradFn *)fn, batchnorm2d_apply, NULL, next_fns, next_fn_count, "BatchNorm2dBackward");
    fn->input = input;
    fn->gamma = gamma;
    fn->batch_mean = batch_mean;
    fn->batch_var = batch_var;
    fn->eps = eps;
    return (GradFn *)fn;
}
