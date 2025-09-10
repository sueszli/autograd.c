#include "ops.h"
#include "../utils/defer.h"
#include "../utils/types.h"
#include "broadcast.h"
#include "tensor.h"
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    TENSOR_OP_ADD,
    TENSOR_OP_SUB,
    TENSOR_OP_MUL,
    TENSOR_OP_DIV,
} tensor_op_type_t;

static bool shapes_match(const tensor_t *a, const tensor_t *b) {
    if (a->ndim != b->ndim) {
        return false;
    }
    for (i32 i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }
    return true;
}

static void reduce_gradient_if_needed(tensor_t *grad, const tensor_t *original_tensor) {
    if (shapes_match(grad, original_tensor)) {
        return;
    }

    u64 grad_size = tensor_size(grad);
    u64 original_size = tensor_size(original_tensor);
    f32 *reduced_data = (f32 *)calloc(original_size, sizeof(f32));
    assert(reduced_data != NULL);

    for (u64 i = 0; i < grad_size; i++) {
        reduced_data[i % original_size] += grad->data[i];
    }

    free(grad->data);
    grad->data = reduced_data;

    free(grad->shape);
    grad->shape = (i32 *)malloc((size_t)original_tensor->ndim * sizeof(i32));
    assert(grad->shape != NULL);
    memcpy(grad->shape, original_tensor->shape, (size_t)original_tensor->ndim * sizeof(i32));
    grad->ndim = original_tensor->ndim;
}

static void accumulate_gradient(tensor_t *tensor, tensor_t *grad_update, bool needs_broadcasting) {
    if (!tensor->requires_grad) {
        return;
    }

    if (tensor->grad == NULL) {
        tensor->grad = tensor_create(NULL, grad_update->shape, grad_update->ndim, false);
        tensor_zero_grad(tensor);
    }

    for (u64 i = 0; i < tensor_size(grad_update); i++) {
        tensor->grad->data[i] += grad_update->data[i];
    }

    if (needs_broadcasting) {
        reduce_gradient_if_needed(tensor->grad, tensor);
    }
}

static tensor_t *tensor_op_execution(tensor_t *a, tensor_t *b, tensor_op_type_t op_type) {
    assert(a != NULL && b != NULL);

    bool needs_broadcasting = !shapes_match(a, b);
    tensor_t *op_a = a;
    tensor_t *op_b = b;

    if (needs_broadcasting) {
        broadcasted_tensors_t broadcasted = tensor_broadcast(a, b);
        op_a = broadcasted.a;
        op_b = broadcasted.b;
    } else {
        assert(tensor_size(a) == tensor_size(b));
    }

    defer({
        if (op_a != a)
            tensor_destroy(op_a);
        if (op_b != b)
            tensor_destroy(op_b);
    });

    u64 size = tensor_size(op_a);
    f32 *new_data = (f32 *)malloc(size * sizeof(f32));
    assert(new_data != NULL);

    for (u64 i = 0; i < size; i++) {
        switch (op_type) {
        case TENSOR_OP_ADD:
            new_data[i] = op_a->data[i] + op_b->data[i];
            break;
        case TENSOR_OP_SUB:
            new_data[i] = op_a->data[i] - op_b->data[i];
            break;
        case TENSOR_OP_MUL:
            new_data[i] = op_a->data[i] * op_b->data[i];
            break;
        case TENSOR_OP_DIV:
            new_data[i] = op_a->data[i] / op_b->data[i];
            break;
        }
    }

    tensor_t *result = tensor_create(new_data, op_a->shape, op_a->ndim, false);
    free(new_data);

    return result;
}

static void add_backward(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool needs_broadcasting = (bool)(intptr_t)t->ctx[2];

    accumulate_gradient(a, t->grad, needs_broadcasting);
    accumulate_gradient(b, t->grad, needs_broadcasting);
}

static void sub_backward(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool needs_broadcasting = (bool)(intptr_t)t->ctx[2];

    accumulate_gradient(a, t->grad, needs_broadcasting);

    tensor_t *neg_grad = tensor_create(NULL, t->grad->shape, t->grad->ndim, false);
    for (u64 i = 0; i < tensor_size(t->grad); i++) {
        neg_grad->data[i] = -t->grad->data[i];
    }
    accumulate_gradient(b, neg_grad, needs_broadcasting);
    tensor_destroy(neg_grad);
}

static void mul_backward(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool needs_broadcasting = (bool)(intptr_t)t->ctx[2];

    tensor_t *grad_a = tensor_op_execution(t->grad, b, TENSOR_OP_MUL);
    tensor_t *grad_b = tensor_op_execution(t->grad, a, TENSOR_OP_MUL);
    defer({
        tensor_destroy(grad_a);
        tensor_destroy(grad_b);
    });

    accumulate_gradient(a, grad_a, needs_broadcasting);
    accumulate_gradient(b, grad_b, needs_broadcasting);
}

static void div_backward(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool needs_broadcasting = (bool)(intptr_t)t->ctx[2];

    tensor_t *grad_a = tensor_op_execution(t->grad, b, TENSOR_OP_DIV);

    tensor_t *b_squared = tensor_op_execution(b, b, TENSOR_OP_MUL);
    tensor_t *a_mul_grad = tensor_op_execution(a, t->grad, TENSOR_OP_MUL);
    tensor_t *numerator = tensor_op_execution(a_mul_grad, t->grad, TENSOR_OP_MUL);
    tensor_t *grad_b = tensor_op_execution(numerator, b_squared, TENSOR_OP_DIV);
    tensor_t *neg_grad_b = tensor_create(NULL, grad_b->shape, grad_b->ndim, false);
    for (u64 i = 0; i < tensor_size(grad_b); i++) {
        neg_grad_b->data[i] = -grad_b->data[i];
    }

    defer({
        tensor_destroy(grad_a);
        tensor_destroy(b_squared);
        tensor_destroy(a_mul_grad);
        tensor_destroy(numerator);
        tensor_destroy(grad_b);
        tensor_destroy(neg_grad_b);
    });

    accumulate_gradient(a, grad_a, needs_broadcasting);
    accumulate_gradient(b, neg_grad_b, needs_broadcasting);
}

static tensor_t *tensor_op(tensor_t *a, tensor_t *b, tensor_op_type_t op_type) {
    tensor_t *result = tensor_op_execution(a, b, op_type);

    bool requires_grad = a->requires_grad || b->requires_grad;
    if (requires_grad) {
        result->requires_grad = true;
        result->ctx = (void **)malloc(3 * sizeof(void *));
        assert(result->ctx != NULL);
        result->ctx[0] = a;
        result->ctx[1] = b;
        bool needs_broadcasting = !shapes_match(a, b);
        result->ctx[2] = (void *)(intptr_t)needs_broadcasting;
        result->ctx_size = 3;

        switch (op_type) {
        case TENSOR_OP_ADD:
            result->grad_fn = add_backward;
            break;
        case TENSOR_OP_SUB:
            result->grad_fn = sub_backward;
            break;
        case TENSOR_OP_MUL:
            result->grad_fn = mul_backward;
            break;
        case TENSOR_OP_DIV:
            result->grad_fn = div_backward;
            break;
        }
    }

    return result;
}

tensor_t *tensor_add(tensor_t *a, tensor_t *b) { return tensor_op(a, b, TENSOR_OP_ADD); }
tensor_t *tensor_sub(tensor_t *a, tensor_t *b) { return tensor_op(a, b, TENSOR_OP_SUB); }
tensor_t *tensor_mul(tensor_t *a, tensor_t *b) { return tensor_op(a, b, TENSOR_OP_MUL); }
tensor_t *tensor_div(tensor_t *a, tensor_t *b) { return tensor_op(a, b, TENSOR_OP_DIV); }
