#include "ops.h"
#include "../utils/defer.h"
#include "../utils/types.h"
#include "broadcast.h"
#include "tensor.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static tensor_t *ensure_gradient_tensor(tensor_t *tensor, i32 *shape, i32 ndim) {
    assert(tensor != NULL && shape != NULL);
    if (tensor->grad == NULL) {
        tensor->grad = tensor_create(NULL, shape, ndim, false);
        assert(tensor->grad != NULL);
        memset(tensor->grad->data, 0, tensor_size(tensor->grad) * sizeof(f32));
    }
    return tensor->grad;
}

static void reduce_gradient_if_needed(tensor_t *grad_tensor, tensor_t *original_tensor) {
    assert(grad_tensor != NULL && original_tensor != NULL);

    u64 grad_size = tensor_size(grad_tensor);
    u64 orig_size = tensor_size(original_tensor);

    if (grad_size == orig_size && grad_tensor->ndim == original_tensor->ndim) {
        bool shapes_match = true;
        for (i32 i = 0; i < grad_tensor->ndim; i++) {
            if (grad_tensor->shape[i] != original_tensor->shape[i]) {
                shapes_match = false;
                break;
            }
        }
        if (shapes_match)
            return;
    }

    f32 *reduced_data = (f32 *)calloc(orig_size, sizeof(f32));
    assert(reduced_data != NULL);

    for (u64 i = 0; i < grad_size; i++) {
        reduced_data[i % orig_size] += grad_tensor->data[i];
    }

    free(grad_tensor->data);
    grad_tensor->data = reduced_data;

    free(grad_tensor->shape);
    grad_tensor->shape = (i32 *)malloc((size_t)original_tensor->ndim * sizeof(i32));
    assert(grad_tensor->shape != NULL);
    memcpy(grad_tensor->shape, original_tensor->shape, (size_t)original_tensor->ndim * sizeof(i32));
    grad_tensor->ndim = original_tensor->ndim;
}

typedef void (*gradient_accumulator_fn)(u64 i, tensor_t *grad_tensor, tensor_t *output_grad, tensor_t *broadcast_a, tensor_t *broadcast_b);
static void add_gradient(u64 i, tensor_t *grad_tensor, tensor_t *output_grad, tensor_t *broadcast_a __attribute__((unused)), tensor_t *broadcast_b __attribute__((unused))) { grad_tensor->data[i] += output_grad->data[i]; }
static void sub_gradient_a(u64 i, tensor_t *grad_tensor, tensor_t *output_grad, tensor_t *broadcast_a __attribute__((unused)), tensor_t *broadcast_b __attribute__((unused))) { grad_tensor->data[i] += output_grad->data[i]; }
static void sub_gradient_b(u64 i, tensor_t *grad_tensor, tensor_t *output_grad, tensor_t *broadcast_a __attribute__((unused)), tensor_t *broadcast_b __attribute__((unused))) { grad_tensor->data[i] -= output_grad->data[i]; }
static void mul_gradient_a(u64 i, tensor_t *grad_tensor, tensor_t *output_grad, tensor_t *broadcast_a __attribute__((unused)), tensor_t *broadcast_b) { grad_tensor->data[i] += output_grad->data[i] * broadcast_b->data[i]; }
static void mul_gradient_b(u64 i, tensor_t *grad_tensor, tensor_t *output_grad, tensor_t *broadcast_a, tensor_t *broadcast_b __attribute__((unused))) { grad_tensor->data[i] += output_grad->data[i] * broadcast_a->data[i]; }
static void div_gradient_a(u64 i, tensor_t *grad_tensor, tensor_t *output_grad, tensor_t *broadcast_a __attribute__((unused)), tensor_t *broadcast_b) { grad_tensor->data[i] += output_grad->data[i] / broadcast_b->data[i]; }
static void div_gradient_b(u64 i, tensor_t *grad_tensor, tensor_t *output_grad, tensor_t *broadcast_a, tensor_t *broadcast_b) { grad_tensor->data[i] -= output_grad->data[i] * broadcast_a->data[i] / (broadcast_b->data[i] * broadcast_b->data[i]); }

static void accumulate_gradient(tensor_t *tensor, tensor_t *output_tensor, bool use_broadcasting, tensor_t *broadcast_a, tensor_t *broadcast_b, gradient_accumulator_fn accumulator) {
    assert(tensor != NULL && output_tensor != NULL && broadcast_a != NULL && broadcast_b != NULL && accumulator != NULL);
    if (!tensor->requires_grad)
        return;

    ensure_gradient_tensor(tensor, output_tensor->shape, output_tensor->ndim);

    u64 size = tensor_size(output_tensor);
    for (u64 i = 0; i < size; i++) {
        accumulator(i, tensor->grad, output_tensor->grad, broadcast_a, broadcast_b);
    }

    if (use_broadcasting) {
        reduce_gradient_if_needed(tensor->grad, tensor);
    }
}

static void add_backward_broadcast(tensor_t *t) {
    assert(t != NULL && t->ctx != NULL);
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool used_broadcasting = (bool)(intptr_t)t->ctx[2];

    accumulate_gradient(a, t, used_broadcasting, a, b, add_gradient);
    accumulate_gradient(b, t, used_broadcasting, a, b, add_gradient);
}

static void sub_backward_broadcast(tensor_t *t) {
    assert(t != NULL && t->ctx != NULL);
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool used_broadcasting = (bool)(intptr_t)t->ctx[2];

    accumulate_gradient(a, t, used_broadcasting, a, b, sub_gradient_a);
    accumulate_gradient(b, t, used_broadcasting, a, b, sub_gradient_b);
}

static void mul_backward_broadcast(tensor_t *t) {
    assert(t != NULL && t->ctx != NULL);
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool used_broadcasting = (bool)(intptr_t)t->ctx[2];

    broadcasted_tensors_t broadcasted = tensor_broadcast(a, b);
    assert(broadcasted.a != NULL && broadcasted.b != NULL);
    defer({
        if (broadcasted.a != a)
            tensor_destroy(broadcasted.a);
        if (broadcasted.b != b)
            tensor_destroy(broadcasted.b);
    });

    accumulate_gradient(a, t, used_broadcasting, broadcasted.a, broadcasted.b, mul_gradient_a);
    accumulate_gradient(b, t, used_broadcasting, broadcasted.a, broadcasted.b, mul_gradient_b);
}

static void div_backward_broadcast(tensor_t *t) {
    assert(t != NULL && t->ctx != NULL);
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool used_broadcasting = (bool)(intptr_t)t->ctx[2];

    broadcasted_tensors_t broadcasted = tensor_broadcast(a, b);
    assert(broadcasted.a != NULL && broadcasted.b != NULL);
    defer({
        if (broadcasted.a != a)
            tensor_destroy(broadcasted.a);
        if (broadcasted.b != b)
            tensor_destroy(broadcasted.b);
    });

    accumulate_gradient(a, t, used_broadcasting, broadcasted.a, broadcasted.b, div_gradient_a);
    accumulate_gradient(b, t, used_broadcasting, broadcasted.a, broadcasted.b, div_gradient_b);
}

static tensor_t *create_result_tensor(tensor_t *a, tensor_t *b, tensor_t *result_a, tensor_op_t op, bool use_broadcasting) {
    assert(a != NULL && b != NULL && result_a != NULL);
    bool requires_grad = a->requires_grad || b->requires_grad;
    tensor_t *result = tensor_create(NULL, result_a->shape, result_a->ndim, requires_grad);
    assert(result != NULL);

    if (requires_grad) {
        result->ctx = (void **)malloc(3 * sizeof(void *));
        assert(result->ctx != NULL);

        result->ctx[0] = a;
        result->ctx[1] = b;
        result->ctx[2] = (void *)(intptr_t)use_broadcasting;
        result->ctx_size = 3;

        switch (op) {
        case TENSOR_OP_ADD:
            result->grad_fn = add_backward_broadcast;
            break;
        case TENSOR_OP_SUB:
            result->grad_fn = sub_backward_broadcast;
            break;
        case TENSOR_OP_MUL:
            result->grad_fn = mul_backward_broadcast;
            break;
        case TENSOR_OP_DIV:
            result->grad_fn = div_backward_broadcast;
            break;
        default:
            assert(false && "invalid operation type");
        }
    }

    return result;
}

static tensor_t *perform_elementwise_op(tensor_t *a, tensor_t *b, tensor_op_t op, bool use_broadcasting) {
    assert(a != NULL && b != NULL);

    tensor_t *op_a = a;
    tensor_t *op_b = b;

    if (use_broadcasting) {
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
        switch (op) {
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
            assert(op_b->data[i] != 0.0f);
            new_data[i] = op_a->data[i] / op_b->data[i];
            break;
        default:
            assert(false && "invalid operation type");
        }
    }

    tensor_t *result = create_result_tensor(a, b, op_a, op, use_broadcasting && (op_a != a));
    free(result->data);
    result->data = new_data;

    return result;
}

tensor_t *tensor_op_add(tensor_t *a, tensor_t *b, bool use_broadcasting) { return perform_elementwise_op(a, b, TENSOR_OP_ADD, use_broadcasting); }

tensor_t *tensor_op_sub(tensor_t *a, tensor_t *b, bool use_broadcasting) { return perform_elementwise_op(a, b, TENSOR_OP_SUB, use_broadcasting); }

tensor_t *tensor_op_mul(tensor_t *a, tensor_t *b, bool use_broadcasting) { return perform_elementwise_op(a, b, TENSOR_OP_MUL, use_broadcasting); }

tensor_t *tensor_op_div(tensor_t *a, tensor_t *b, bool use_broadcasting) { return perform_elementwise_op(a, b, TENSOR_OP_DIV, use_broadcasting); }

tensor_t *tensor_op_generic(tensor_t *a, tensor_t *b, tensor_op_t op, bool use_broadcasting) {
    assert(a != NULL && b != NULL);
    return perform_elementwise_op(a, b, op, use_broadcasting);
}
