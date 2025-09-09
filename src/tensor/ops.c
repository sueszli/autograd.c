#include "ops.h"
#include "../utils/types.h"
#include "broadcast.h"
#include "tensor.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static bool validate_tensor_shapes(tensor_t *a, tensor_t *b) {
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

static tensor_t *ensure_gradient_tensor(tensor_t *tensor, i32 *shape, i32 ndim) {
    if (tensor->grad == NULL) {
        tensor->grad = tensor_create(NULL, shape, ndim, false);
        assert(tensor->grad != NULL);
        memset(tensor->grad->data, 0, tensor_size(tensor->grad) * sizeof(f32));
    }
    return tensor->grad;
}

static void setup_broadcast_tensors(tensor_t *a, tensor_t *b, bool use_broadcasting, tensor_t **broadcast_a, tensor_t **broadcast_b, bool *need_free_a, bool *need_free_b) {
    *broadcast_a = a;
    *broadcast_b = b;
    *need_free_a = false;
    *need_free_b = false;

    if (use_broadcasting) {
        shape_t broadcast_shape = get_tensor_broadcast_shape(a, b);
        if (broadcast_shape.shape) {
            *broadcast_a = tensor_broadcast_to(a, broadcast_shape.shape, broadcast_shape.ndim);
            *broadcast_b = tensor_broadcast_to(b, broadcast_shape.shape, broadcast_shape.ndim);
            if (*broadcast_a && *broadcast_b) {
                *need_free_a = true;
                *need_free_b = true;
            }
            shape_free(&broadcast_shape);
        }
    }
}

static void cleanup_broadcast_tensors(tensor_t *broadcast_a, tensor_t *broadcast_b, bool need_free_a, bool need_free_b) {
    if (need_free_a)
        tensor_destroy(broadcast_a);
    if (need_free_b)
        tensor_destroy(broadcast_b);
}

static void reduce_gradient_if_needed(tensor_t *grad_tensor, tensor_t *original_tensor) {
    if (!grad_tensor || !original_tensor)
        return;

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
    if (!reduced_data)
        return;

    for (u64 i = 0; i < grad_size && i < orig_size; i++) {
        reduced_data[i % orig_size] += grad_tensor->data[i];
    }

    free(grad_tensor->data);
    grad_tensor->data = reduced_data;

    free(grad_tensor->shape);
    grad_tensor->shape = (i32 *)malloc((size_t)original_tensor->ndim * sizeof(i32));
    if (grad_tensor->shape) {
        memcpy(grad_tensor->shape, original_tensor->shape, (size_t)original_tensor->ndim * sizeof(i32));
        grad_tensor->ndim = original_tensor->ndim;
    }
}

typedef void (*gradient_accumulator_fn)(u64 i, tensor_t *grad_tensor, tensor_t *output_grad, tensor_t *broadcast_a, tensor_t *broadcast_b);

static void add_gradient(u64 i, tensor_t *grad_tensor, tensor_t *output_grad, tensor_t *broadcast_a, tensor_t *broadcast_b) {
    (void)broadcast_a;
    (void)broadcast_b;
    grad_tensor->data[i] += output_grad->data[i];
}

static void sub_gradient_a(u64 i, tensor_t *grad_tensor, tensor_t *output_grad, tensor_t *broadcast_a, tensor_t *broadcast_b) {
    (void)broadcast_a;
    (void)broadcast_b;
    grad_tensor->data[i] += output_grad->data[i];
}

static void sub_gradient_b(u64 i, tensor_t *grad_tensor, tensor_t *output_grad, tensor_t *broadcast_a, tensor_t *broadcast_b) {
    (void)broadcast_a;
    (void)broadcast_b;
    grad_tensor->data[i] -= output_grad->data[i];
}

static void mul_gradient_a(u64 i, tensor_t *grad_tensor, tensor_t *output_grad, tensor_t *broadcast_a, tensor_t *broadcast_b) {
    (void)broadcast_a;
    grad_tensor->data[i] += output_grad->data[i] * broadcast_b->data[i];
}

static void mul_gradient_b(u64 i, tensor_t *grad_tensor, tensor_t *output_grad, tensor_t *broadcast_a, tensor_t *broadcast_b) {
    (void)broadcast_b;
    grad_tensor->data[i] += output_grad->data[i] * broadcast_a->data[i];
}

static void div_gradient_a(u64 i, tensor_t *grad_tensor, tensor_t *output_grad, tensor_t *broadcast_a, tensor_t *broadcast_b) {
    (void)broadcast_a;
    grad_tensor->data[i] += output_grad->data[i] / broadcast_b->data[i];
}

static void div_gradient_b(u64 i, tensor_t *grad_tensor, tensor_t *output_grad, tensor_t *broadcast_a, tensor_t *broadcast_b) {
    grad_tensor->data[i] -= output_grad->data[i] * broadcast_a->data[i] / (broadcast_b->data[i] * broadcast_b->data[i]);
}

static void accumulate_gradient(tensor_t *tensor, tensor_t *output_tensor, bool use_broadcasting, tensor_t *broadcast_a, tensor_t *broadcast_b, gradient_accumulator_fn accumulator) {
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
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool used_broadcasting = (bool)(intptr_t)t->ctx[2];

    accumulate_gradient(a, t, used_broadcasting, a, b, add_gradient);
    accumulate_gradient(b, t, used_broadcasting, a, b, add_gradient);
}

static void sub_backward_broadcast(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool used_broadcasting = (bool)(intptr_t)t->ctx[2];

    accumulate_gradient(a, t, used_broadcasting, a, b, sub_gradient_a);
    accumulate_gradient(b, t, used_broadcasting, a, b, sub_gradient_b);
}

static void mul_backward_broadcast(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool used_broadcasting = (bool)(intptr_t)t->ctx[2];

    tensor_t *broadcast_a, *broadcast_b;
    bool need_free_a, need_free_b;

    setup_broadcast_tensors(a, b, used_broadcasting, &broadcast_a, &broadcast_b, &need_free_a, &need_free_b);

    accumulate_gradient(a, t, used_broadcasting, broadcast_a, broadcast_b, mul_gradient_a);
    accumulate_gradient(b, t, used_broadcasting, broadcast_a, broadcast_b, mul_gradient_b);

    cleanup_broadcast_tensors(broadcast_a, broadcast_b, need_free_a, need_free_b);
}

static void div_backward_broadcast(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool used_broadcasting = (bool)(intptr_t)t->ctx[2];

    tensor_t *broadcast_a, *broadcast_b;
    bool need_free_a, need_free_b;

    setup_broadcast_tensors(a, b, used_broadcasting, &broadcast_a, &broadcast_b, &need_free_a, &need_free_b);

    accumulate_gradient(a, t, used_broadcasting, broadcast_a, broadcast_b, div_gradient_a);
    accumulate_gradient(b, t, used_broadcasting, broadcast_a, broadcast_b, div_gradient_b);

    cleanup_broadcast_tensors(broadcast_a, broadcast_b, need_free_a, need_free_b);
}

static tensor_t *create_result_tensor(tensor_t *a, tensor_t *b, tensor_t *result_a, tensor_op_t op, bool use_broadcasting) {
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

    tensor_t *result_a = a;
    tensor_t *result_b = b;
    bool need_free_a = false, need_free_b = false;

    if (use_broadcasting) {
        if (!tensor_can_broadcast(a, b)) {
            return NULL;
        }

        shape_t broadcast_shape = get_tensor_broadcast_shape(a, b);
        if (!broadcast_shape.shape) {
            return NULL;
        }

        result_a = tensor_broadcast_to(a, broadcast_shape.shape, broadcast_shape.ndim);
        result_b = tensor_broadcast_to(b, broadcast_shape.shape, broadcast_shape.ndim);
        need_free_a = true;
        need_free_b = true;

        shape_free(&broadcast_shape);

        if (!result_a || !result_b) {
            if (need_free_a && result_a)
                tensor_destroy(result_a);
            if (need_free_b && result_b)
                tensor_destroy(result_b);
            return NULL;
        }
    } else {
        if (!validate_tensor_shapes(a, b)) {
            return NULL;
        }
    }

    u64 size = tensor_size(result_a);
    f32 *new_data = (f32 *)malloc(size * sizeof(f32));
    assert(new_data != NULL);

    switch (op) {
    case TENSOR_OP_ADD:
        for (u64 i = 0; i < size; i++) {
            new_data[i] = result_a->data[i] + result_b->data[i];
        }
        break;
    case TENSOR_OP_SUB:
        for (u64 i = 0; i < size; i++) {
            new_data[i] = result_a->data[i] - result_b->data[i];
        }
        break;
    case TENSOR_OP_MUL:
        for (u64 i = 0; i < size; i++) {
            new_data[i] = result_a->data[i] * result_b->data[i];
        }
        break;
    case TENSOR_OP_DIV:
        for (u64 i = 0; i < size; i++) {
            if (result_b->data[i] == 0.0f) {
                free(new_data);
                if (need_free_a)
                    tensor_destroy(result_a);
                if (need_free_b)
                    tensor_destroy(result_b);
                return NULL;
            }
            new_data[i] = result_a->data[i] / result_b->data[i];
        }
        break;
    default:
        assert(false && "invalid operation type");
    }

    tensor_t *result = create_result_tensor(a, b, result_a, op, use_broadcasting);

    memcpy(result->data, new_data, size * sizeof(f32));
    free(new_data);

    if (need_free_a)
        tensor_destroy(result_a);
    if (need_free_b)
        tensor_destroy(result_b);

    return result;
}

tensor_t *tensor_op_add(tensor_t *a, tensor_t *b, bool use_broadcasting) {
    if (!a || !b)
        return NULL;
    return perform_elementwise_op(a, b, TENSOR_OP_ADD, use_broadcasting);
}

tensor_t *tensor_op_sub(tensor_t *a, tensor_t *b, bool use_broadcasting) {
    if (!a || !b)
        return NULL;
    return perform_elementwise_op(a, b, TENSOR_OP_SUB, use_broadcasting);
}

tensor_t *tensor_op_mul(tensor_t *a, tensor_t *b, bool use_broadcasting) {
    if (!a || !b)
        return NULL;
    return perform_elementwise_op(a, b, TENSOR_OP_MUL, use_broadcasting);
}

tensor_t *tensor_op_div(tensor_t *a, tensor_t *b, bool use_broadcasting) {
    if (!a || !b)
        return NULL;
    return perform_elementwise_op(a, b, TENSOR_OP_DIV, use_broadcasting);
}

tensor_t *tensor_op_generic(tensor_t *a, tensor_t *b, tensor_op_t op, bool use_broadcasting) {
    if (!a || !b)
        return NULL;
    return perform_elementwise_op(a, b, op, use_broadcasting);
}