#include "ops.h"
#include "broadcast.h"
#include "tensor.h"
#include "../utils/types.h"
#include <stdlib.h>
#include <string.h>

static void add_backward_broadcast(tensor_t *t);
static void sub_backward_broadcast(tensor_t *t);
static void mul_backward_broadcast(tensor_t *t);
static void div_backward_broadcast(tensor_t *t);

static tensor_t *perform_elementwise_op(tensor_t *a, tensor_t *b, tensor_op_t op, bool use_broadcasting) {
    if (!a || !b) return NULL;
    
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
            if (result_a && need_free_a) tensor_destroy(result_a);
            if (result_b && need_free_b) tensor_destroy(result_b);
            return NULL;
        }
    } else {
        if (a->ndim != b->ndim) {
            return NULL;
        }
        for (i32 i = 0; i < a->ndim; i++) {
            if (a->shape[i] != b->shape[i]) {
                return NULL;
            }
        }
    }
    
    u64 size = tensor_size(result_a);
    f32 *new_data = (f32 *)malloc(size * sizeof(f32));
    if (!new_data) {
        if (need_free_a) tensor_destroy(result_a);
        if (need_free_b) tensor_destroy(result_b);
        return NULL;
    }
    
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
                    if (need_free_a) tensor_destroy(result_a);
                    if (need_free_b) tensor_destroy(result_b);
                    return NULL;
                }
                new_data[i] = result_a->data[i] / result_b->data[i];
            }
            break;
        default:
            free(new_data);
            if (need_free_a) tensor_destroy(result_a);
            if (need_free_b) tensor_destroy(result_b);
            return NULL;
    }
    
    bool requires_grad = a->requires_grad || b->requires_grad;
    tensor_t *result = tensor_create(new_data, result_a->shape, result_a->ndim, requires_grad);
    free(new_data);
    
    if (!result) {
        if (need_free_a) tensor_destroy(result_a);
        if (need_free_b) tensor_destroy(result_b);
        return NULL;
    }
    
    if (requires_grad) {
        result->ctx = (void **)malloc(3 * sizeof(void *));
        result->ctx[0] = a; // original tensors for gradient computation
        result->ctx[1] = b;
        result->ctx[2] = (void *)(intptr_t)use_broadcasting; // store broadcasting flag
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
        }
    }
    
    if (need_free_a) tensor_destroy(result_a);
    if (need_free_b) tensor_destroy(result_b);
    
    return result;
}

static void reduce_gradient_if_needed(tensor_t *grad_tensor, tensor_t *original_tensor) {
    if (!grad_tensor || !original_tensor) return;
    
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
        if (shapes_match) return;
    }
    
    f32 *reduced_data = (f32 *)calloc(orig_size, sizeof(f32));
    if (!reduced_data) return;
    
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

static void add_backward_broadcast(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool used_broadcasting = (bool)(intptr_t)t->ctx[2];
    
    if (a->requires_grad) {
        if (a->grad == NULL) {
            a->grad = tensor_create(NULL, t->shape, t->ndim, false);
            memset(a->grad->data, 0, tensor_size(a->grad) * sizeof(f32));
        }
        
        for (u64 i = 0; i < tensor_size(t); i++) {
            a->grad->data[i] += t->grad->data[i];
        }
        
        if (used_broadcasting) {
            reduce_gradient_if_needed(a->grad, a);
        }
    }
    
    if (b->requires_grad) {
        if (b->grad == NULL) {
            b->grad = tensor_create(NULL, t->shape, t->ndim, false);
            memset(b->grad->data, 0, tensor_size(b->grad) * sizeof(f32));
        }
        
        for (u64 i = 0; i < tensor_size(t); i++) {
            b->grad->data[i] += t->grad->data[i];
        }
        
        if (used_broadcasting) {
            reduce_gradient_if_needed(b->grad, b);
        }
    }
}

static void sub_backward_broadcast(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool used_broadcasting = (bool)(intptr_t)t->ctx[2];
    
    if (a->requires_grad) {
        if (a->grad == NULL) {
            a->grad = tensor_create(NULL, t->shape, t->ndim, false);
            memset(a->grad->data, 0, tensor_size(a->grad) * sizeof(f32));
        }
        
        for (u64 i = 0; i < tensor_size(t); i++) {
            a->grad->data[i] += t->grad->data[i];
        }
        
        if (used_broadcasting) {
            reduce_gradient_if_needed(a->grad, a);
        }
    }
    
    if (b->requires_grad) {
        if (b->grad == NULL) {
            b->grad = tensor_create(NULL, t->shape, t->ndim, false);
            memset(b->grad->data, 0, tensor_size(b->grad) * sizeof(f32));
        }
        
        for (u64 i = 0; i < tensor_size(t); i++) {
            b->grad->data[i] -= t->grad->data[i]; // negative gradient for subtraction
        }
        
        if (used_broadcasting) {
            reduce_gradient_if_needed(b->grad, b);
        }
    }
}

static void mul_backward_broadcast(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool used_broadcasting = (bool)(intptr_t)t->ctx[2];
    
    tensor_t *broadcast_a = a;
    tensor_t *broadcast_b = b;
    bool need_free_a = false, need_free_b = false;
    
    if (used_broadcasting) {
        shape_t broadcast_shape = get_tensor_broadcast_shape(a, b);
        if (broadcast_shape.shape) {
            broadcast_a = tensor_broadcast_to(a, broadcast_shape.shape, broadcast_shape.ndim);
            broadcast_b = tensor_broadcast_to(b, broadcast_shape.shape, broadcast_shape.ndim);
            need_free_a = true;
            need_free_b = true;
            shape_free(&broadcast_shape);
        }
    }
    
    if (a->requires_grad) {
        if (a->grad == NULL) {
            a->grad = tensor_create(NULL, t->shape, t->ndim, false);
            memset(a->grad->data, 0, tensor_size(a->grad) * sizeof(f32));
        }
        
        for (u64 i = 0; i < tensor_size(t); i++) {
            a->grad->data[i] += t->grad->data[i] * broadcast_b->data[i];
        }
        
        if (used_broadcasting) {
            reduce_gradient_if_needed(a->grad, a);
        }
    }
    
    if (b->requires_grad) {
        if (b->grad == NULL) {
            b->grad = tensor_create(NULL, t->shape, t->ndim, false);
            memset(b->grad->data, 0, tensor_size(b->grad) * sizeof(f32));
        }
        
        for (u64 i = 0; i < tensor_size(t); i++) {
            b->grad->data[i] += t->grad->data[i] * broadcast_a->data[i];
        }
        
        if (used_broadcasting) {
            reduce_gradient_if_needed(b->grad, b);
        }
    }
    
    if (need_free_a) tensor_destroy(broadcast_a);
    if (need_free_b) tensor_destroy(broadcast_b);
}

static void div_backward_broadcast(tensor_t *t) {
    tensor_t *a = (tensor_t *)t->ctx[0];
    tensor_t *b = (tensor_t *)t->ctx[1];
    bool used_broadcasting = (bool)(intptr_t)t->ctx[2];
    
    tensor_t *broadcast_a = a;
    tensor_t *broadcast_b = b;
    bool need_free_a = false, need_free_b = false;
    
    if (used_broadcasting) {
        shape_t broadcast_shape = get_tensor_broadcast_shape(a, b);
        if (broadcast_shape.shape) {
            broadcast_a = tensor_broadcast_to(a, broadcast_shape.shape, broadcast_shape.ndim);
            broadcast_b = tensor_broadcast_to(b, broadcast_shape.shape, broadcast_shape.ndim);
            need_free_a = true;
            need_free_b = true;
            shape_free(&broadcast_shape);
        }
    }
    
    if (a->requires_grad) {
        if (a->grad == NULL) {
            a->grad = tensor_create(NULL, t->shape, t->ndim, false);
            memset(a->grad->data, 0, tensor_size(a->grad) * sizeof(f32));
        }
        
        for (u64 i = 0; i < tensor_size(t); i++) {
            a->grad->data[i] += t->grad->data[i] / broadcast_b->data[i];
        }
        
        if (used_broadcasting) {
            reduce_gradient_if_needed(a->grad, a);
        }
    }
    
    if (b->requires_grad) {
        if (b->grad == NULL) {
            b->grad = tensor_create(NULL, t->shape, t->ndim, false);
            memset(b->grad->data, 0, tensor_size(b->grad) * sizeof(f32));
        }
        
        for (u64 i = 0; i < tensor_size(t); i++) {
            b->grad->data[i] -= t->grad->data[i] * broadcast_a->data[i] / (broadcast_b->data[i] * broadcast_b->data[i]);
        }
        
        if (used_broadcasting) {
            reduce_gradient_if_needed(b->grad, b);
        }
    }
    
    if (need_free_a) tensor_destroy(broadcast_a);
    if (need_free_b) tensor_destroy(broadcast_b);
}

tensor_t *tensor_op_add(tensor_t *a, tensor_t *b, bool use_broadcasting) {
    return perform_elementwise_op(a, b, TENSOR_OP_ADD, use_broadcasting);
}

tensor_t *tensor_op_sub(tensor_t *a, tensor_t *b, bool use_broadcasting) {
    return perform_elementwise_op(a, b, TENSOR_OP_SUB, use_broadcasting);
}

tensor_t *tensor_op_mul(tensor_t *a, tensor_t *b, bool use_broadcasting) {
    return perform_elementwise_op(a, b, TENSOR_OP_MUL, use_broadcasting);
}

tensor_t *tensor_op_div(tensor_t *a, tensor_t *b, bool use_broadcasting) {
    return perform_elementwise_op(a, b, TENSOR_OP_DIV, use_broadcasting);
}

tensor_t *tensor_op_generic(tensor_t *a, tensor_t *b, tensor_op_t op, bool use_broadcasting) {
    return perform_elementwise_op(a, b, op, use_broadcasting);
}