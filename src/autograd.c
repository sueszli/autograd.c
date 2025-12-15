#include "autograd.h"
#include "tensor.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =========================================================================
// Helpers
// =========================================================================

static void accumulate_grad(Tensor *t, Tensor *grad) {
    if (!t || !grad || !t->requires_grad) return;
    if (t->grad == NULL) {
        t->grad = tensor_create(grad->data, grad->shape, grad->ndim, false);
    } else {
        Tensor *new_grad = tensor_add(t->grad, grad);
        tensor_free(t->grad);
        t->grad = new_grad;
    }
}

// =========================================================================
// Topological Sort
// =========================================================================

typedef struct {
    GradFn **arr;
    int size;
    int capacity;
} PtrArray;

static void ptr_array_append(PtrArray *arr, GradFn *fn) {
    if (arr->size >= arr->capacity) {
        arr->capacity = (arr->capacity == 0) ? 16 : arr->capacity * 2;
        arr->arr = (GradFn **)realloc(arr->arr, arr->capacity * sizeof(GradFn *));
        assert(arr->arr != NULL);
    }
    arr->arr[arr->size++] = fn;
}

static bool ptr_array_contains(PtrArray *arr, GradFn *fn) {
    for (int i = 0; i < arr->size; i++) {
        if (arr->arr[i] == fn) return true;
    }
    return false;
}

static void build_topo_recursive(GradFn *fn, PtrArray *topo, PtrArray *visited) {
    if (ptr_array_contains(visited, fn)) return;
    ptr_array_append(visited, fn);

    for (int i = 0; i < fn->num_next; i++) {
        if (fn->next_fns[i]) {
            build_topo_recursive(fn->next_fns[i], topo, visited);
        }
    }
    ptr_array_append(topo, fn);
}

// =========================================================================
// Autograd Engine
// =========================================================================

void backward(Tensor *root, Tensor *grad) {
    assert(root != NULL);

    // 1. Seed gradient
    if (root->grad == NULL) {
        if (grad == NULL) {
            if (root->size == 1) {
                uint64_t shape[] = {1};
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

    if (!root->grad_fn) return;

    // 2. Topological sort
    PtrArray topo = {0};
    PtrArray visited = {0};
    build_topo_recursive(root->grad_fn, &topo, &visited);

    // 3. Backward pass
    for (int i = topo.size - 1; i >= 0; i--) {
        GradFn *fn = topo.arr[i];
        
        Tensor *out = fn->out_tensor;
        // out might be NULL if the tensor was freed? 
        // Assumption: User keeps tensors alive.
        if (out && out->grad) {
             fn->apply(fn, out->grad);
        }
    }
    
    if (topo.arr) free(topo.arr);
    if (visited.arr) free(visited.arr);
}

void grad_fn_init(GradFn *fn, void (*apply)(GradFn *, struct Tensor *), GradFn **next_fns, int num_next, char *name) {
    fn->apply = apply;
    fn->next_fns = next_fns;
    fn->num_next = num_next;
    fn->name = name ? strdup(name) : NULL;
    fn->out_tensor = NULL; 
}

void grad_fn_free(GradFn *fn) {
    if (!fn) return;
    if (fn->next_fns) free(fn->next_fns);
    if (fn->name) free(fn->name);
}

// =========================================================================
// Operations Backwards
// =========================================================================

// ADD
typedef struct {
    GradFn base;
    Tensor *a; // input a
    Tensor *b; // input b
} AddBackward;

static void add_apply(GradFn *base, Tensor *grad_output) {
    AddBackward *self = (AddBackward *)base;
    accumulate_grad(self->a, grad_output);
    accumulate_grad(self->b, grad_output);
}

GradFn *new_add_backward(Tensor *a, Tensor *b) {
    AddBackward *fn = malloc(sizeof(AddBackward));
    GradFn **next_fns = malloc(2 * sizeof(GradFn *));
    int count = 0;
    if (a->grad_fn) next_fns[count++] = a->grad_fn;
    if (b->grad_fn) next_fns[count++] = b->grad_fn;
    
    grad_fn_init((GradFn *)fn, add_apply, next_fns, count, "AddBackward");
    fn->a = a;
    fn->b = b;
    return (GradFn *)fn;
}

// SUB
typedef struct {
    GradFn base;
    Tensor *a; // input a
    Tensor *b; // input b
} SubBackward;

static void sub_apply(GradFn *base, Tensor *grad_output) {
    SubBackward *self = (SubBackward *)base;
    accumulate_grad(self->a, grad_output);
    
    // grad_b = -grad_output
    // create -1 tensor
    // or tensor_neg helper? let's multiply by scalar -1
    // Simplest: 0 - grad_output
    
    // Use tensor_mul with scalar -1 tensor
    // (Assuming we have tensor_scalar op or broadcasting)
    // Quick hack: tensor_zeros_like(grad) - grad
    uint64_t *shape = malloc(grad_output->ndim * sizeof(uint64_t));
    memcpy(shape, grad_output->shape, grad_output->ndim * sizeof(uint64_t));
    Tensor *zeros = tensor_zeros(shape, grad_output->ndim, false);
    free(shape);
    Tensor *neg_grad = tensor_sub(zeros, grad_output);
    tensor_free(zeros);
    
    accumulate_grad(self->b, neg_grad);
    tensor_free(neg_grad);
}

GradFn *new_sub_backward(Tensor *a, Tensor *b) {
    SubBackward *fn = malloc(sizeof(SubBackward));
    GradFn **next_fns = malloc(2 * sizeof(GradFn *));
    int count = 0;
    if (a->grad_fn) next_fns[count++] = a->grad_fn;
    if (b->grad_fn) next_fns[count++] = b->grad_fn;

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

static void mul_apply(GradFn *base, Tensor *grad_output) {
    MulBackward *self = (MulBackward *)base;
    
    // da = grad * b
    Tensor *da = tensor_mul(grad_output, self->b);
    accumulate_grad(self->a, da);
    tensor_free(da);
    
    // db = grad * a
    Tensor *db = tensor_mul(grad_output, self->a);
    accumulate_grad(self->b, db);
    tensor_free(db);
}

GradFn *new_mul_backward(Tensor *a, Tensor *b) {
    MulBackward *fn = malloc(sizeof(MulBackward));
    GradFn **next_fns = malloc(2 * sizeof(GradFn *));
    int count = 0;
    if (a->grad_fn) next_fns[count++] = a->grad_fn;
    if (b->grad_fn) next_fns[count++] = b->grad_fn;

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

static void div_apply(GradFn *base, Tensor *grad_output) {
    DivBackward *self = (DivBackward *)base;
    
    // da = grad * (1/b) = grad / b
    Tensor *da = tensor_div(grad_output, self->b);
    accumulate_grad(self->a, da);
    tensor_free(da);
    
    // db = grad * (-a / b^2) = -grad * a / (b*b)
    // -grad
    uint64_t *shape = malloc(grad_output->ndim * sizeof(uint64_t));
    memcpy(shape, grad_output->shape, grad_output->ndim * sizeof(uint64_t));
    Tensor *zeros = tensor_zeros(shape, grad_output->ndim, false);
    free(shape);
    Tensor *neg_grad = tensor_sub(zeros, grad_output);
    tensor_free(zeros);
    
    Tensor *num = tensor_mul(neg_grad, self->a);
    tensor_free(neg_grad);
    
    Tensor *denom = tensor_mul(self->b, self->b);
    Tensor *db = tensor_div(num, denom);
    tensor_free(num);
    tensor_free(denom);
    
    accumulate_grad(self->b, db);
    tensor_free(db);
}

GradFn *new_div_backward(Tensor *a, Tensor *b) {
    DivBackward *fn = malloc(sizeof(DivBackward));
    GradFn **next_fns = malloc(2 * sizeof(GradFn *));
    int count = 0;
    if (a->grad_fn) next_fns[count++] = a->grad_fn;
    if (b->grad_fn) next_fns[count++] = b->grad_fn;

    grad_fn_init((GradFn *)fn, div_apply, next_fns, count, "DivBackward");
    fn->a = a;
    fn->b = b;
    return (GradFn *)fn;
}
