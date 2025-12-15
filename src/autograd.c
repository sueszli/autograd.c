#include "autograd.h"
#include "tensor.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_RECURSION_DEPTH 1024
#define MAX_NDIM 32

//
// Helpers
//

static void accumulate_grad(Tensor *t, const Tensor *grad) {
    if (!t || !grad || !t->requires_grad) {
        return;
    }
    assert(grad->data != NULL);

    if (t->grad == NULL) {
        Tensor *zeros = tensor_zeros(t->shape, t->ndim, false);
        t->grad = tensor_add(zeros, grad);
        tensor_free(zeros);
    } else {
        Tensor *new_grad = tensor_add(t->grad, grad);
        tensor_free(t->grad);
        t->grad = new_grad;
    }
}

//
// Topological Sort
//

typedef struct {
    GradFn **arr;
    int32_t size;
    int32_t capacity;
} PtrArray;

static void ptr_array_init(PtrArray *arr) {
    arr->size = 0;
    arr->capacity = 16;
    arr->arr = (GradFn **)malloc((size_t)arr->capacity * sizeof(GradFn *));
    assert(arr->arr != NULL && "malloc failed");
}

static void ptr_array_free(PtrArray *arr) {
    if (arr->arr) {
        free(arr->arr);
        arr->arr = NULL;
    }
}

static void ptr_array_append(PtrArray *arr, GradFn *fn) {
    assert(arr != NULL);
    assert(fn != NULL);
    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        GradFn **new_arr = (GradFn **)realloc(arr->arr, (size_t)arr->capacity * sizeof(GradFn *));
        assert(new_arr != NULL && "realloc failed");
        arr->arr = new_arr;
    }
    arr->arr[arr->size++] = fn;
}

static bool ptr_array_contains(const PtrArray *arr, const GradFn *fn) {
    assert(arr != NULL);
    for (int32_t i = 0; i < arr->size; i++) {
        if (arr->arr[i] == fn) {
            return true;
        }
    }
    return false;
}

static void build_topo_recursive(GradFn *fn, PtrArray *topo, PtrArray *visited, int32_t depth) {
    assert(depth < MAX_RECURSION_DEPTH && "Recursion depth exceeded");
    assert(fn != NULL);

    if (ptr_array_contains(visited, fn)) {
        return;
    }
    ptr_array_append(visited, fn);

    for (int32_t i = 0; i < fn->num_next; i++) {
        if (fn->next_fns[i]) {
            build_topo_recursive(fn->next_fns[i], topo, visited, depth + 1);
        }
    }
    ptr_array_append(topo, fn);
}

//
// Autograd Engine
//

void backward(Tensor *root, const Tensor *grad) {
    assert(root != NULL);

    // 1. Seed gradient
    if (root->grad == NULL) {
        if (grad == NULL) {
            bool is_scalar = (root->size == 1);
            if (is_scalar) {
                const uint64_t shape[] = {1};
                root->grad = tensor_create(NULL, shape, 0, false);
                root->grad->data[0] = 1.0f;
            } else {
                assert(false && "grad must be specified for non-scalar root");
            }
        } else {
            root->grad = tensor_create(grad->data, grad->shape, grad->ndim, false);
        }
    } else {
        if (grad != NULL) {
            accumulate_grad(root, grad);
        }
    }

    if (!root->grad_fn) {
        return;
    }

    // 2. Topological sort
    PtrArray topo;
    ptr_array_init(&topo);
    PtrArray visited;
    ptr_array_init(&visited);

    build_topo_recursive(root->grad_fn, &topo, &visited, 0);

    // 3. Backward pass
    for (int32_t i = topo.size - 1; i >= 0; i--) {
        GradFn *fn = topo.arr[i];
        assert(fn != NULL);

        Tensor *out = fn->out_tensor;
        if (out && out->grad) {
            fn->apply(fn, out->grad);
        }
    }

    ptr_array_free(&topo);
    ptr_array_free(&visited);
}

void grad_fn_init(GradFn *fn, void (*apply)(GradFn *, const struct Tensor *), GradFn **next_fns, int num_next, const char *name) {
    assert(fn != NULL);
    assert(apply != NULL);

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

static Tensor *unbroadcast(const Tensor *grad, const Tensor *input) {
    if (!grad || !input)
        return NULL;

    // 1. Reduce rank
    Tensor *curr = NULL;

    bool made_copy = false;

    if (grad->ndim > input->ndim) {
        Tensor *next = tensor_sum(grad, 0, false);
        curr = next;
        made_copy = true;

        while (curr->ndim > input->ndim) {
            next = tensor_sum(curr, 0, false);
            tensor_free(curr);
            curr = next;
        }
    } else {
        curr = (Tensor *)grad; // Cast const away temporarily, but we treat it as read-only until we copy
    }

    for (uint64_t i = 0; i < input->ndim; i++) {
        // safely handle curr->shape vs input->shape
        // curr->ndim should equal input->ndim here because of step 1.

        // However, if we didn't do step 1, curr == grad.

        // In PyTorch, if input is (1, 3) and grad is (2, 3), sum(0) makes it (1, 3) -> (3) ? No sum(0) removes dim 0?
        // tensor_sum with keepdims=false reduces rank.
        // If input (3,) grad (2, 3). sum(0) -> (3,). Match.

        // Wait, if input has shape (1, 3) and grad (2, 3).
        // Grad has rank 2. Input rank 2.
        // input->shape[0] == 1. grad->shape[0] == 2.
        // We need to sum dim 0, keeping dim.

        if (input->shape[i] == 1 && curr->shape[i] > 1) {
            Tensor *next = tensor_sum(curr, (int64_t)i, true);
            if (made_copy) {
                tensor_free(curr);
            }
            curr = next;
            made_copy = true;
        }
    }

    if (!made_copy) {
        // Return a clone so caller always owns the result
        return tensor_create(grad->data, grad->shape, grad->ndim, false);
    }

    return curr;
}

static void accumulate_grad_unbroadcast(Tensor *t, const Tensor *grad) {
    if (!t || !grad || !t->requires_grad)
        return;

    Tensor *adj_grad = unbroadcast(grad, t);
    if (adj_grad) {
        accumulate_grad(t, adj_grad);
        tensor_free(adj_grad);
    }
}

// ADD
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
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(2 * sizeof(GradFn *));
    assert(next_fns != NULL);

    int count = 0;
    if (a->grad_fn)
        next_fns[count++] = a->grad_fn;
    if (b->grad_fn)
        next_fns[count++] = b->grad_fn;

    grad_fn_init((GradFn *)fn, add_apply, next_fns, count, "AddBackward");
    fn->a = a;
    fn->b = b;
    return (GradFn *)fn;
}

// SUB
typedef struct {
    GradFn base;
    Tensor *a;
    Tensor *b;
} SubBackward;

static void sub_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    SubBackward *self = (SubBackward *)base;
    accumulate_grad_unbroadcast(self->a, grad_output);

    // grad_b = -grad_output
    uint64_t *shape = (uint64_t *)malloc(grad_output->ndim * sizeof(uint64_t));
    assert(shape != NULL);
    memcpy(shape, grad_output->shape, grad_output->ndim * sizeof(uint64_t));

    Tensor *zeros = tensor_zeros(shape, grad_output->ndim, false);
    free(shape);

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

    int count = 0;
    if (a->grad_fn)
        next_fns[count++] = a->grad_fn;
    if (b->grad_fn)
        next_fns[count++] = b->grad_fn;

    grad_fn_init((GradFn *)fn, sub_apply, next_fns, count, "SubBackward");
    fn->a = a;
    fn->b = b;
    return (GradFn *)fn;
}

// MUL
typedef struct {
    GradFn base;
    Tensor *a;
    Tensor *b;
} MulBackward;

static void mul_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    MulBackward *self = (MulBackward *)base;

    Tensor *da = tensor_mul(grad_output, self->b);
    accumulate_grad_unbroadcast(self->a, da);
    tensor_free(da);

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
    int count = 0;
    if (a->grad_fn)
        next_fns[count++] = a->grad_fn;
    if (b->grad_fn)
        next_fns[count++] = b->grad_fn;

    grad_fn_init((GradFn *)fn, mul_apply, next_fns, count, "MulBackward");
    fn->a = a;
    fn->b = b;
    return (GradFn *)fn;
}

// DIV
typedef struct {
    GradFn base;
    Tensor *a;
    Tensor *b;
} DivBackward;

static void div_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    DivBackward *self = (DivBackward *)base;

    Tensor *da = tensor_div(grad_output, self->b);
    accumulate_grad_unbroadcast(self->a, da);
    tensor_free(da);

    uint64_t *shape = (uint64_t *)malloc(grad_output->ndim * sizeof(uint64_t));
    assert(shape != NULL);
    memcpy(shape, grad_output->shape, grad_output->ndim * sizeof(uint64_t));

    Tensor *zeros = tensor_zeros(shape, grad_output->ndim, false);
    free(shape);

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
    int count = 0;
    if (a->grad_fn)
        next_fns[count++] = a->grad_fn;
    if (b->grad_fn)
        next_fns[count++] = b->grad_fn;

    grad_fn_init((GradFn *)fn, div_apply, next_fns, count, "DivBackward");
    fn->a = a;
    fn->b = b;
    return (GradFn *)fn;
}

// MATMUL
typedef struct {
    GradFn base;
    Tensor *a;
    Tensor *b;
} MatmulBackward;

static void matmul_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    MatmulBackward *self = (MatmulBackward *)base;

    Tensor *b_T = tensor_transpose(self->b, 0, 1);
    Tensor *da = tensor_matmul(grad_output, b_T);
    accumulate_grad(self->a, da);
    tensor_free(b_T);
    tensor_free(da);

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
    int count = 0;
    if (a->grad_fn)
        next_fns[count++] = a->grad_fn;
    if (b->grad_fn)
        next_fns[count++] = b->grad_fn;

    grad_fn_init((GradFn *)fn, matmul_apply, next_fns, count, "MatmulBackward");
    fn->a = a;
    fn->b = b;
    return (GradFn *)fn;
}

// SUM
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

    const Tensor *grad_expanded = grad_output;
    bool needs_free = false;

    if (!self->keepdims) {
        int64_t ndim = (int64_t)self->input->ndim;
        int64_t new_shape[MAX_NDIM] = {0};
        int g = 0;
        int64_t target_dim = (self->dim_idx < 0) ? (self->dim_idx + ndim) : self->dim_idx;

        for (int64_t i = 0; i < ndim; i++) {
            if (i == target_dim) {
                new_shape[i] = 1;
            } else {
                new_shape[i] = (int64_t)grad_output->shape[g++];
            }
        }

        grad_expanded = tensor_reshape(grad_output, new_shape, (uint64_t)ndim);
        needs_free = true;
    }

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
    int count = 0;
    if (input->grad_fn)
        next_fns[count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, sum_apply, next_fns, count, "SumBackward");
    fn->input = input;
    fn->dim_idx = dim_idx;
    fn->keepdims = keepdims;
    return (GradFn *)fn;
}

// RELU
typedef struct {
    GradFn base;
    Tensor *input;
} ReluBackward;

static void relu_apply(GradFn *base, const Tensor *grad_output) {
    assert(base != NULL);
    assert(grad_output != NULL);
    ReluBackward *self = (ReluBackward *)base;

    assert(grad_output->size == self->input->size && "ReLU grad shape mismatch");
    Tensor *grad_input = tensor_zeros(grad_output->shape, grad_output->ndim, false);

    for (uint64_t i = 0; i < grad_output->size; i++) {
        if (self->input->data[i] > 0.0f) {
            grad_input->data[i] = grad_output->data[i];
        } else {
            grad_input->data[i] = 0.0f;
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
    int count = 0;
    if (input->grad_fn)
        next_fns[count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, relu_apply, next_fns, count, "ReluBackward");
    fn->input = input;
    return (GradFn *)fn;
}

// SIGMOID
typedef struct {
    GradFn base;
    Tensor *input;
    Tensor *output;
} SigmoidBackward;

static void sigmoid_apply(GradFn *base, const Tensor *grad_output) {
    SigmoidBackward *self = (SigmoidBackward *)base;

    Tensor *ones = tensor_zeros(self->output->shape, self->output->ndim, false);
    for (uint64_t i = 0; i < ones->size; i++)
        ones->data[i] = 1.0f;

    Tensor *one_minus_out = tensor_sub(ones, self->output);
    tensor_free(ones);

    Tensor *d_sigmoid = tensor_mul(self->output, one_minus_out);
    tensor_free(one_minus_out);

    Tensor *grad_input = tensor_mul(grad_output, d_sigmoid);
    tensor_free(d_sigmoid);

    accumulate_grad(self->input, grad_input);
    tensor_free(grad_input);
}

GradFn *new_sigmoid_backward(Tensor *input, Tensor *output) {
    SigmoidBackward *fn = (SigmoidBackward *)malloc(sizeof(SigmoidBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);
    int count = 0;
    if (input->grad_fn)
        next_fns[count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, sigmoid_apply, next_fns, count, "SigmoidBackward");
    fn->input = input;
    fn->output = output;
    return (GradFn *)fn;
}

// SOFTMAX
typedef struct {
    GradFn base;
    Tensor *input;
    Tensor *output;
    int64_t dim;
} SoftmaxBackward;

static void softmax_apply(GradFn *base, const Tensor *grad_output) {
    SoftmaxBackward *self = (SoftmaxBackward *)base;

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
    SoftmaxBackward *fn = (SoftmaxBackward *)malloc(sizeof(SoftmaxBackward));
    assert(fn != NULL);

    GradFn **next_fns = (GradFn **)malloc(sizeof(GradFn *));
    assert(next_fns != NULL);
    int count = 0;
    if (input->grad_fn)
        next_fns[count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, softmax_apply, next_fns, count, "SoftmaxBackward");
    fn->input = input;
    fn->output = output;
    fn->dim = dim;
    return (GradFn *)fn;
}

// RESHAPE
typedef struct {
    GradFn base;
    Tensor *input;
    uint64_t *old_shape;
    uint64_t old_ndim;
} ReshapeBackward;

static void reshape_apply(GradFn *base, const Tensor *grad_output) {
    ReshapeBackward *self = (ReshapeBackward *)base;
    assert(base != NULL);
    assert(grad_output != NULL);

    int64_t signed_shape[MAX_NDIM] = {0};
    for (uint64_t i = 0; i < self->old_ndim; i++) {
        signed_shape[i] = (int64_t)self->old_shape[i];
    }

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
    int count = 0;
    if (input->grad_fn)
        next_fns[count++] = input->grad_fn;

    grad_fn_init((GradFn *)fn, reshape_apply, next_fns, count, "ReshapeBackward");
    fn->input = input;
    fn->old_ndim = old_ndim;
    fn->old_shape = (uint64_t *)malloc(old_ndim * sizeof(uint64_t));
    assert(fn->old_shape != NULL);
    memcpy(fn->old_shape, old_shape, old_ndim * sizeof(uint64_t));

    return (GradFn *)fn;
}
