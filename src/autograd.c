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
 * If the tensor does not require gradients, this function does nothing.
 *
 * t:    The tensor to update.
 * grad: The gradient to accumulate (must be same shape/size unless broadcasting logic handled elsewhere,
 *       but here we assume direct addition is valid or `tensor_add` handles it).
 */
static void accumulate_grad(Tensor *t, const Tensor *grad) {
    // If t is NULL, it's likely a logic error in the caller (e.g. missing input tensor in context)
    assert(t != NULL);

    // If we don't need gradients for this tensor, skip.
    if (!t->requires_grad) {
        return;
    }

    assert(grad != NULL);
    assert(grad->data != NULL || grad->size == 0);

    if (t->grad == NULL) {
        // First gradient arrival: initialize as zero and add (or just copy)
        // cloning via zeros + add is safe and ensures we own the memory
        Tensor *zeros = tensor_zeros(t->shape, t->ndim, false);
        t->grad = tensor_add(zeros, grad);
        tensor_free(zeros);
    } else {
        // Accumulate: new_grad = old_grad + grad
        Tensor *new_grad = tensor_add(t->grad, grad);
        tensor_free(t->grad);
        t->grad = new_grad;
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
    uint64_t size;
    uint64_t capacity;
} PtrArray;

static void ptr_array_init(PtrArray *arr) {
    assert(arr != NULL);
    arr->size = 0;
    arr->capacity = 16;
    arr->data = (GradFn **)malloc((uint64_t)arr->capacity * sizeof(GradFn *));
    assert(arr->data != NULL && "malloc failed");
}

static void ptr_array_free(PtrArray *arr) {
    assert(arr != NULL);
    if (arr->data) {
        free(arr->data);
        arr->data = NULL;
    }
    arr->size = 0;
    arr->capacity = 0;
}

static void ptr_array_append(PtrArray *arr, GradFn *fn) {
    assert(arr != NULL);
    assert(fn != NULL);

    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        GradFn **new_data = (GradFn **)realloc(arr->data, (uint64_t)arr->capacity * sizeof(GradFn *));
        assert(new_data != NULL && "realloc failed");
        arr->data = new_data;
    }
    arr->data[arr->size++] = fn;
}

static bool ptr_array_contains(const PtrArray *arr, const GradFn *fn) {
    assert(arr != NULL);
    assert(arr->size < MAX_ARRAY_SIZE && "Array size exceeds maximum limit");
    for (uint64_t fn_idx = 0; fn_idx < arr->size; fn_idx++) {
        if (arr->data[fn_idx] == fn) {
            return true;
        }
    }
    return false;
}

/*
 * Recursive helper for topological sort.
 * Performs a Depth First Search (DFS) to order nodes.
 */
static void build_topo_recursive(GradFn *fn, PtrArray *topo, PtrArray *visited, uint64_t depth) {
    assert(depth < MAX_RECURSION_DEPTH && "Recursion depth exceeded: graph too deep");
    assert(fn != NULL);
    assert(topo != NULL);
    assert(visited != NULL);

    if (ptr_array_contains(visited, fn)) {
        return;
    }
    ptr_array_append(visited, fn);

    for (int64_t next_idx = 0; next_idx < fn->num_next; next_idx++) {
        if (fn->next_fns[next_idx]) {
            build_topo_recursive(fn->next_fns[next_idx], topo, visited, depth + 1);
        }
    }
    ptr_array_append(topo, fn);
}

//
// Autograd Engine
//

/*
 * backward: Computes the gradient of current tensor w.r.t. graph leaves.
 *
 * 1. Seeds the gradient (d_output = 1.0 if scalar, or uses provided grad).
 * 2. Builds a topological order of the compute graph efficiently.
 * 3. Iterates in reverse topological order, calling .apply() on each node.
 */
void backward(Tensor *root, const Tensor *grad) {
    assert(root != NULL);

    // 1. Seed gradient
    if (root->grad == NULL) {
        if (grad == NULL) {
            // Implicit scalar gradient
            if (root->size == 1) {
                const uint64_t shape[] = {1};
                root->grad = tensor_create(NULL, shape, 0, false);
                root->grad->data[0] = 1.0f;
            } else {
                assert(false && "Grad must be specified for non-scalar root");
            }
        } else {
            // Explicit gradient provided
            assert(grad->data != NULL || grad->size == 0);

            // Explicitly copy the gradient to ensure we own it.
            // This is safer than relying on implicit semantics.
            root->grad = tensor_create(grad->data, grad->shape, grad->ndim, false);
        }
    } else {
        // If gradient already exists, accumulate.
        if (grad != NULL) {
            accumulate_grad(root, grad);
        }
    }

    if (!root->grad_fn) {
        // Leaf node, nothing to propagate
        return;
    }

    // 2. Topological sort
    PtrArray topo;
    ptr_array_init(&topo);
    PtrArray visited;
    ptr_array_init(&visited);

    build_topo_recursive(root->grad_fn, &topo, &visited, 0);

    // 3. Backward pass
    // Process nodes in reverse topological order (children before parents)
    for (int64_t i = (int64_t)topo.size - 1; i >= 0; i--) {
        GradFn *fn = topo.data[i];
        assert(fn != NULL);

        Tensor *out = fn->out_tensor;
        // If the output of this function has a gradient, propagate it
        if (out && out->grad) {
            fn->apply(fn, out->grad);
        }
    }

    ptr_array_free(&topo);
    ptr_array_free(&visited);
}

void grad_fn_init(GradFn *fn, void (*apply)(GradFn *, const struct Tensor *), GradFn **next_fns, int64_t num_next, const char *name) {
    assert(fn != NULL);
    assert(apply != NULL);
    // next_fns can be NULL if num_next is 0

    fn->apply = apply;
    fn->next_fns = next_fns;
    fn->num_next = num_next;
    fn->name = name ? strdup(name) : NULL;
    fn->out_tensor = NULL;
}

void grad_fn_free(GradFn *fn) {
    if (!fn) {
        return;
    }
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
 * Handle broadcasting in backward pass.
 * If the input shape was smaller than result shape (due to broadcasting),
 * we must sum up the gradients along the broadcasted dimensions.
 *
 * Example:
 *   Forward: A (2, 1) + B (3) -> C (2, 3)
 *   Backward: dL/dC (2, 3)
 *   dL/dA = sum(dL/dC, axis=1) -> (2, 1)
 *   dL/dB = sum(dL/dC, axis=0) -> (1, 3) -> reshape -> (3)
 */
static Tensor *unbroadcast(const Tensor *grad, const Tensor *input) {
    if (!grad || !input) {
        return NULL;
    }

    const Tensor *curr = grad;
    bool needs_free = false; // Do we own 'curr'?

    // 1. Collapse extra dimensions (e.g. valid for (2,3) -> (3) cases? Wait.)
    // Broadcasting adds dimensions on the left.
    // So if grad is (N, C, H, W) and input is (C, H, W), we sum first axis.

    while (curr->ndim > input->ndim) {
        Tensor *next = tensor_sum(curr, 0, false);
        if (needs_free) {
            tensor_free((Tensor *)curr);
        }
        curr = next;
        needs_free = true;
    }

    // 2. Collapse broadcasted dimensions (where dimension was 1)
    // Now ndim should be equal.
    assert(curr->ndim == input->ndim);
    assert(input->ndim < MAX_NDIM && "Number of dimensions exceeds maximum");

    for (uint64_t dim_idx = 0; dim_idx < input->ndim; dim_idx++) {
        // If input has size 1 but gradient has size > 1, it was broadcasted.
        if (input->shape[dim_idx] == 1 && curr->shape[dim_idx] > 1) {
            Tensor *next = tensor_sum(curr, (int64_t)dim_idx, true);
            if (needs_free) {
                tensor_free((Tensor *)curr);
            }
            curr = next;
            needs_free = true;
        }
    }

    // If we haven't made a copy yet, we must clone because the caller expects to own the result.
    if (!needs_free) {
        return tensor_create(grad->data, grad->shape, grad->ndim, false);
    }

    return (Tensor *)curr;
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

    grad_fn_init((GradFn *)fn, add_apply, next_fns, next_fn_count, "AddBackward");
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

    grad_fn_init((GradFn *)fn, sub_apply, next_fns, next_fn_count, "SubBackward");
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

    grad_fn_init((GradFn *)fn, mul_apply, next_fns, next_fn_count, "MulBackward");
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

    grad_fn_init((GradFn *)fn, div_apply, next_fns, next_fn_count, "DivBackward");
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

    grad_fn_init((GradFn *)fn, matmul_apply, next_fns, next_fn_count, "MatmulBackward");
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

    grad_fn_init((GradFn *)fn, sum_apply, next_fns, next_fn_count, "SumBackward");
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

    grad_fn_init((GradFn *)fn, relu_apply, next_fns, next_fn_count, "ReluBackward");
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

    grad_fn_init((GradFn *)fn, sigmoid_apply, next_fns, next_fn_count, "SigmoidBackward");
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

    grad_fn_init((GradFn *)fn, softmax_apply, next_fns, next_fn_count, "SoftmaxBackward");
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

    grad_fn_init((GradFn *)fn, reshape_apply, next_fns, next_fn_count, "ReshapeBackward");
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

    grad_fn_init((GradFn *)fn, transpose_apply, next_fns, next_fn_count, "TransposeBackward");
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

    grad_fn_init((GradFn *)fn, getitem_apply, next_fns, next_fn_count, "GetItemBackward");
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

    grad_fn_init((GradFn *)fn, gelu_apply, next_fns, next_fn_count, "GELUBackward");
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

    // grad_output IS A SCALAR TENSOR (loss is scalar)
    assert(grad_output->size == 1);
    float32_t g = grad_output->data[0];

    Tensor *grad_pred = tensor_create(NULL, self->predictions->shape, self->predictions->ndim, false);
    assert(self->predictions->size < MAX_TENSOR_SIZE && "Tensor size exceeds maximum limit");
    float32_t sample_count_f32 = (float32_t)self->predictions->size;

    for (uint64_t elem_idx = 0; elem_idx < self->predictions->size; elem_idx++) {
        // grad = 2 * (pred - target) / N
        float32_t diff = self->predictions->data[elem_idx] - self->targets->data[elem_idx];
        grad_pred->data[elem_idx] = g * 2.0f * diff / sample_count_f32;
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

    grad_fn_init((GradFn *)fn, mse_apply, next_fns, next_fn_count, "MSEBackward");
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
    float32_t g = grad_output->data[0];

    Tensor *grad_pred = tensor_create(NULL, self->predictions->shape, self->predictions->ndim, false);
    assert(self->predictions->size < MAX_TENSOR_SIZE && "Tensor size exceeds maximum limit");
    float32_t sample_count_f32 = (float32_t)self->predictions->size;

    for (uint64_t elem_idx = 0; elem_idx < self->predictions->size; elem_idx++) {
        float32_t p = self->predictions->data[elem_idx];
        float32_t y = self->targets->data[elem_idx];

        if (p < EPSILON)
            p = EPSILON;
        if (p > 1.0f - EPSILON)
            p = 1.0f - EPSILON;

        // grad = (p - y) / (p * (1 - p) * N)
        // For BCE Loss = -mean(y*log(p) + (1-y)*log(1-p))
        // dLoss/dp = (p - y) / (p * (1-p)) / N
        // Multiplied by incoming gradient g

        float32_t denom = p * (1.0f - p) * sample_count_f32;
        grad_pred->data[elem_idx] = g * (p - y) / denom;
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

    grad_fn_init((GradFn *)fn, bce_apply, next_fns, next_fn_count, "BCEBackward");
    fn->predictions = predictions;
    fn->targets = targets;
    return (GradFn *)fn;
}

// --- CrossEntropy ---
typedef struct {
    GradFn base;
    Tensor *logits;
    Tensor *targets;
} CrossEntropyBackward;

static void crossentropy_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    CrossEntropyBackward *self = (CrossEntropyBackward *)base;

    assert(grad_output->size == 1);
    float32_t g = grad_output->data[0];

    // Recompute softmax
    assert(self->logits->ndim == 2);
    uint64_t batch_size = self->logits->shape[0];
    uint64_t class_count = self->logits->shape[1];
    assert(batch_size < MAX_TENSOR_SIZE && "Batch size exceeds maximum limit");
    assert(class_count < MAX_TENSOR_SIZE && "Class count exceeds maximum limit");

    Tensor *grad_logits = tensor_create(NULL, self->logits->shape, self->logits->ndim, false);

    for (uint64_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // 1. max for stability
        float32_t max_val = -INFINITY;
        for (uint64_t class_idx = 0; class_idx < class_count; class_idx++) {
            float32_t val = self->logits->data[batch_idx * class_count + class_idx];
            if (val > max_val)
                max_val = val;
        }

        // 2. exp sum
        float32_t sum_exp = 0.0f;
        for (uint64_t class_idx = 0; class_idx < class_count; class_idx++) {
            sum_exp += expf(self->logits->data[batch_idx * class_count + class_idx] - max_val);
        }

        // 3. compute grad
        // grad_logit[j] = (probs[j] - indicator[j==target]) / batch_size
        uint64_t target_class_idx = (uint64_t)self->targets->data[batch_idx];

        for (uint64_t class_idx = 0; class_idx < class_count; class_idx++) {
            float32_t prob = expf(self->logits->data[batch_idx * class_count + class_idx] - max_val) / sum_exp;
            float32_t indicator = (class_idx == target_class_idx) ? 1.0f : 0.0f;

            grad_logits->data[batch_idx * class_count + class_idx] = g * (prob - indicator) / (float32_t)batch_size;
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

    grad_fn_init((GradFn *)fn, crossentropy_apply, next_fns, next_fn_count, "CrossEntropyBackward");
    fn->logits = logits;
    fn->targets = targets;
    return (GradFn *)fn;
}
