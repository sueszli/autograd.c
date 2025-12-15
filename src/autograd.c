#include "autograd.h"
#include "tensor.h"
#include "tensor_backward.h"
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
#define MAX_NDIM 32

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
    AddBackward *self = (AddBackward *)base;
    Tensor *grad_a = NULL;
    Tensor *grad_b = NULL;
    tensor_add_backward(grad_output, self->a, self->b, &grad_a, &grad_b);
    if(grad_a) {
        accumulate_grad(self->a, grad_a);
        tensor_free(grad_a);
    }
    if(grad_b) {
        accumulate_grad(self->b, grad_b);
        tensor_free(grad_b);
    }
}

GradFn *new_add_backward(Tensor *a, Tensor *b) {
    AddBackward *fn = (AddBackward *)malloc(sizeof(AddBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(2 * sizeof(GradFn *));
    int64_t cnt = 0;
    if(a->grad_fn) next_fns[cnt++] = a->grad_fn;
    if(b->grad_fn) next_fns[cnt++] = b->grad_fn;
    grad_fn_init((GradFn *)fn, add_apply, NULL, next_fns, cnt, "AddBackward");
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
    SubBackward *self = (SubBackward *)base;
    Tensor *grad_a = NULL;
    Tensor *grad_b = NULL;
    tensor_sub_backward(grad_output, self->a, self->b, &grad_a, &grad_b);
    if(grad_a) {
        accumulate_grad(self->a, grad_a);
        tensor_free(grad_a);
    }
    if(grad_b) {
        accumulate_grad(self->b, grad_b);
        tensor_free(grad_b);
    }
}

GradFn *new_sub_backward(Tensor *a, Tensor *b) {
    SubBackward *fn = (SubBackward *)malloc(sizeof(SubBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(2 * sizeof(GradFn *));
    int64_t cnt = 0;
    if(a->grad_fn) next_fns[cnt++] = a->grad_fn;
    if(b->grad_fn) next_fns[cnt++] = b->grad_fn;
    grad_fn_init((GradFn *)fn, sub_apply, NULL, next_fns, cnt, "SubBackward");
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
    MulBackward *self = (MulBackward *)base;
    Tensor *grad_a = NULL;
    Tensor *grad_b = NULL;
    tensor_mul_backward(grad_output, self->a, self->b, &grad_a, &grad_b);
    if(grad_a) {
        accumulate_grad(self->a, grad_a);
        tensor_free(grad_a);
    }
    if(grad_b) {
        accumulate_grad(self->b, grad_b);
        tensor_free(grad_b);
    }
}

GradFn *new_mul_backward(Tensor *a, Tensor *b) {
    MulBackward *fn = (MulBackward *)malloc(sizeof(MulBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(2 * sizeof(GradFn *));
    int64_t cnt = 0;
    if(a->grad_fn) next_fns[cnt++] = a->grad_fn;
    if(b->grad_fn) next_fns[cnt++] = b->grad_fn;
    grad_fn_init((GradFn *)fn, mul_apply, NULL, next_fns, cnt, "MulBackward");
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
    DivBackward *self = (DivBackward *)base;
    Tensor *grad_a = NULL;
    Tensor *grad_b = NULL;
    tensor_div_backward(grad_output, self->a, self->b, &grad_a, &grad_b);
    if(grad_a) {
        accumulate_grad(self->a, grad_a);
        tensor_free(grad_a);
    }
    if(grad_b) {
        accumulate_grad(self->b, grad_b);
        tensor_free(grad_b);
    }
}

GradFn *new_div_backward(Tensor *a, Tensor *b) {
    DivBackward *fn = (DivBackward *)malloc(sizeof(DivBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(2 * sizeof(GradFn *));
    int64_t cnt = 0;
    if(a->grad_fn) next_fns[cnt++] = a->grad_fn;
    if(b->grad_fn) next_fns[cnt++] = b->grad_fn;
    grad_fn_init((GradFn *)fn, div_apply, NULL, next_fns, cnt, "DivBackward");
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
    MatmulBackward *self = (MatmulBackward *)base;
    Tensor *grad_a = NULL;
    Tensor *grad_b = NULL;
    tensor_matmul_backward(grad_output, self->a, self->b, &grad_a, &grad_b);
    if(grad_a) {
        accumulate_grad(self->a, grad_a);
        tensor_free(grad_a);
    }
    if(grad_b) {
        accumulate_grad(self->b, grad_b);
        tensor_free(grad_b);
    }
}

GradFn *new_matmul_backward(Tensor *a, Tensor *b) {
    MatmulBackward *fn = (MatmulBackward *)malloc(sizeof(MatmulBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(2 * sizeof(GradFn *));
    int64_t cnt = 0;
    if(a->grad_fn) next_fns[cnt++] = a->grad_fn;
    if(b->grad_fn) next_fns[cnt++] = b->grad_fn;
    grad_fn_init((GradFn *)fn, matmul_apply, NULL, next_fns, cnt, "MatmulBackward");
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
    SumBackward *self = (SumBackward *)base;
    Tensor *grad_input = tensor_sum_backward(grad_output, self->input, self->dim_idx, self->keepdims);
    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

GradFn *new_sum_backward(Tensor *input, int64_t dim_idx, bool keepdims) {
    SumBackward *fn = (SumBackward *)malloc(sizeof(SumBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    int64_t cnt = 0;
    if(input->grad_fn) next_fns[cnt++] = input->grad_fn;
    grad_fn_init((GradFn *)fn, sum_apply, NULL, next_fns, cnt, "SumBackward");
    fn->input = input;
    fn->dim_idx = dim_idx;
    fn->keepdims = keepdims;
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
    MeanBackward *self = (MeanBackward *)base;
    Tensor *grad_input = tensor_mean_backward(grad_output, self->input, self->dim_idx, self->keepdims);
    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

GradFn *new_mean_backward(Tensor *input, int64_t dim_idx, bool keepdims) {
    MeanBackward *fn = (MeanBackward *)malloc(sizeof(MeanBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    int64_t cnt = 0;
    if(input->grad_fn) next_fns[cnt++] = input->grad_fn;
    grad_fn_init((GradFn *)fn, mean_apply, NULL, next_fns, cnt, "MeanBackward");
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
    MaxBackward *self = (MaxBackward *)base;
    Tensor *grad_input = tensor_max_backward(grad_output, self->input, self->output, self->dim_idx, self->keepdims);
    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

GradFn *new_max_backward(Tensor *input, Tensor *output, int64_t dim_idx, bool keepdims) {
    MaxBackward *fn = (MaxBackward *)malloc(sizeof(MaxBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    int64_t cnt = 0;
    if(input->grad_fn) next_fns[cnt++] = input->grad_fn;
    grad_fn_init((GradFn *)fn, max_apply, NULL, next_fns, cnt, "MaxBackward");
    fn->input = input;
    fn->output = output;
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
    ReluBackward *self = (ReluBackward *)base;
    Tensor *grad_input = tensor_relu_backward(grad_output, self->input);
    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

GradFn *new_relu_backward(Tensor *input) {
    ReluBackward *fn = (ReluBackward *)malloc(sizeof(ReluBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    int64_t cnt = 0;
    if(input->grad_fn) next_fns[cnt++] = input->grad_fn;
    grad_fn_init((GradFn *)fn, relu_apply, NULL, next_fns, cnt, "ReluBackward");
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
    SigmoidBackward *self = (SigmoidBackward *)base;
    Tensor *grad_input = tensor_sigmoid_backward(grad_output, self->output);
    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

GradFn *new_sigmoid_backward(Tensor *input, Tensor *output) {
    SigmoidBackward *fn = (SigmoidBackward *)malloc(sizeof(SigmoidBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    int64_t cnt = 0;
    if(input->grad_fn) next_fns[cnt++] = input->grad_fn;
    grad_fn_init((GradFn *)fn, sigmoid_apply, NULL, next_fns, cnt, "SigmoidBackward");
    fn->input = input;
    fn->output = output;
    return (GradFn *)fn;
}

// --- Tanh ---
typedef struct {
    GradFn base;
    Tensor *input;
    Tensor *output;
} TanhBackward;

static void tanh_apply(GradFn *base, const Tensor *grad_output) {
    TanhBackward *self = (TanhBackward *)base;
    Tensor *grad_input = tensor_tanh_backward(grad_output, self->output);
    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

GradFn *new_tanh_backward(Tensor *input, Tensor *output) {
    TanhBackward *fn = (TanhBackward *)malloc(sizeof(TanhBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    int64_t cnt = 0;
    if(input->grad_fn) next_fns[cnt++] = input->grad_fn;
    grad_fn_init((GradFn *)fn, tanh_apply, NULL, next_fns, cnt, "TanhBackward");
    fn->input = input;
    fn->output = output;
    return (GradFn *)fn;
}

// --- GELU ---
typedef struct {
    GradFn base;
    Tensor *input;
} GELUBackward;

static void gelu_apply(GradFn *base, const Tensor *grad_output) {
    GELUBackward *self = (GELUBackward *)base;
    Tensor *grad_input = tensor_gelu_backward(grad_output, self->input);
    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

GradFn *new_gelu_backward(Tensor *input) {
    GELUBackward *fn = (GELUBackward *)malloc(sizeof(GELUBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    int64_t cnt = 0;
    if(input->grad_fn) next_fns[cnt++] = input->grad_fn;
    grad_fn_init((GradFn *)fn, gelu_apply, NULL, next_fns, cnt, "GELUBackward");
    fn->input = input;
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
    SoftmaxBackward *self = (SoftmaxBackward *)base;
    Tensor *grad_input = tensor_softmax_backward(grad_output, self->output, self->dim);
    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

GradFn *new_softmax_backward(Tensor *input, Tensor *output, int64_t dim) {
    SoftmaxBackward *fn = (SoftmaxBackward *)malloc(sizeof(SoftmaxBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    int64_t cnt = 0;
    if(input->grad_fn) next_fns[cnt++] = input->grad_fn;
    grad_fn_init((GradFn *)fn, softmax_apply, NULL, next_fns, cnt, "SoftmaxBackward");
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
    ReshapeBackward *self = (ReshapeBackward *)base;
    Tensor *grad_input = tensor_reshape_backward(grad_output, self->old_shape, self->old_ndim);
    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

static void reshape_destroy(GradFn *fn) {
    ReshapeBackward *self = (ReshapeBackward *)fn;
    if (self->old_shape) free(self->old_shape);
}

GradFn *new_reshape_backward(Tensor *input, const uint64_t *old_shape, uint64_t old_ndim) {
    ReshapeBackward *fn = (ReshapeBackward *)malloc(sizeof(ReshapeBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    int64_t cnt = 0;
    if(input->grad_fn) next_fns[cnt++] = input->grad_fn;
    grad_fn_init((GradFn *)fn, reshape_apply, reshape_destroy, next_fns, cnt, "ReshapeBackward");
    fn->input = input;
    fn->old_ndim = old_ndim;
    fn->old_shape = (uint64_t *)malloc(old_ndim * sizeof(uint64_t));
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
    TransposeBackward *self = (TransposeBackward *)base;
    Tensor *grad_input = tensor_transpose_backward(grad_output, self->dim0, self->dim1);
    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

GradFn *new_transpose_backward(Tensor *input, uint64_t dim0, uint64_t dim1) {
    TransposeBackward *fn = (TransposeBackward *)malloc(sizeof(TransposeBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    int64_t cnt = 0;
    if(input->grad_fn) next_fns[cnt++] = input->grad_fn;
    grad_fn_init((GradFn *)fn, transpose_apply, NULL, next_fns, cnt, "TransposeBackward");
    fn->input = input;
    fn->dim0 = dim0;
    fn->dim1 = dim1;
    return (GradFn *)fn;
}

// --- GetItem ---
typedef struct {
    GradFn base;
    Tensor *input;
    uint64_t *multidim;
} GetItemBackward;

static void getitem_apply(GradFn *base, const Tensor *grad_output) {
    GetItemBackward *self = (GetItemBackward *)base;
    Tensor *grad_input = tensor_getitem_backward(grad_output, self->input, self->multidim);
    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

static void getitem_destroy(GradFn *fn) {
    GetItemBackward *self = (GetItemBackward *)fn;
    if (self->multidim) free(self->multidim);
}

GradFn *new_getitem_backward(Tensor *input, const uint64_t *multidim) {
    GetItemBackward *fn = (GetItemBackward *)malloc(sizeof(GetItemBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    int64_t cnt = 0;
    if(input->grad_fn) next_fns[cnt++] = input->grad_fn;
    grad_fn_init((GradFn *)fn, getitem_apply, getitem_destroy, next_fns, cnt, "GetItemBackward");
    fn->input = input;
    fn->multidim = (uint64_t *)malloc(input->ndim * sizeof(uint64_t));
    memcpy(fn->multidim, multidim, input->ndim * sizeof(uint64_t));
    return (GradFn *)fn;
}

// --- MSE ---
typedef struct {
    GradFn base;
    Tensor *predictions;
    Tensor *targets;
} MSEBackward;

static void mse_apply(GradFn *base, const Tensor *grad_output) {
    MSEBackward *self = (MSEBackward *)base;
    Tensor *grad_pred = tensor_mse_backward(grad_output, self->predictions, self->targets);
    accumulate_grad(self->predictions, grad_pred);
    tensor_free(grad_pred);
}

GradFn *new_mse_backward(Tensor *predictions, Tensor *targets) {
    MSEBackward *fn = (MSEBackward *)malloc(sizeof(MSEBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    int64_t cnt = 0;
    if(predictions->grad_fn) next_fns[cnt++] = predictions->grad_fn;
    grad_fn_init((GradFn *)fn, mse_apply, NULL, next_fns, cnt, "MSEBackward");
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
    BCEBackward *self = (BCEBackward *)base;
    Tensor *grad_pred = tensor_bce_backward(grad_output, self->predictions, self->targets);
    accumulate_grad(self->predictions, grad_pred);
    tensor_free(grad_pred);
}

GradFn *new_bce_backward(Tensor *predictions, Tensor *targets) {
    BCEBackward *fn = (BCEBackward *)malloc(sizeof(BCEBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    int64_t cnt = 0;
    if(predictions->grad_fn) next_fns[cnt++] = predictions->grad_fn;
    grad_fn_init((GradFn *)fn, bce_apply, NULL, next_fns, cnt, "BCEBackward");
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
    CrossEntropyBackward *self = (CrossEntropyBackward *)base;
    Tensor *grad_logits = tensor_crossentropy_backward(grad_output, self->logits, self->targets);
    accumulate_grad(self->logits, grad_logits);
    tensor_free(grad_logits);
}

GradFn *new_crossentropy_backward(Tensor *logits, Tensor *targets) {
    CrossEntropyBackward *fn = (CrossEntropyBackward *)malloc(sizeof(CrossEntropyBackward));
    assert(fn != NULL);
    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    int64_t cnt = 0;
    if(logits->grad_fn) next_fns[cnt++] = logits->grad_fn;
    grad_fn_init((GradFn *)fn, crossentropy_apply, NULL, next_fns, cnt, "CrossEntropyBackward");
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
