#include "ops/arithmetic.h"
#include "autograd.h"
#include "ops/arithmetic_backward.h"
#include "utils/aligned_alloc.h"
#include <assert.h>
#include <stdlib.h>

/*
 * aligns dimensions from the right.
 * compatible if dimensions are equal or one of them is 1.
 *
 * shape_a: [   3, 1]
 * shape_b: [2, 1, 5]
 *           ^  ^  ^
 * out:     [2, 3, 5]
 */
bool broadcast_shapes_mut(const uint64_t *shape_a, uint64_t ndim_a, const uint64_t *shape_b, uint64_t ndim_b, uint64_t *out_shape, uint64_t *out_ndim) {
    assert(ndim_a <= MAX_NDIM);
    assert(ndim_b <= MAX_NDIM);
    assert(out_shape != NULL);
    assert(out_ndim != NULL);

    uint64_t max_ndim = (ndim_a > ndim_b) ? ndim_a : ndim_b;
    assert(max_ndim <= MAX_NDIM);
    *out_ndim = max_ndim;

    int64_t idx_a = (int64_t)ndim_a - 1;
    int64_t idx_b = (int64_t)ndim_b - 1;
    int64_t idx_out = (int64_t)max_ndim - 1;

    assert(idx_out < (int64_t)MAX_NDIM);
    while (idx_out >= 0) {
        uint64_t dim_a = (idx_a >= 0 && shape_a) ? shape_a[idx_a] : 1;
        uint64_t dim_b = (idx_b >= 0 && shape_b) ? shape_b[idx_b] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }

        if (dim_a == 1) {
            out_shape[idx_out] = dim_b;
        } else {
            out_shape[idx_out] = dim_a;
        }

        idx_a--;
        idx_b--;
        idx_out--;
    }
    return true;
}

typedef float32_t (*binary_op_t)(float32_t, float32_t);

Tensor *tensor_binary_op(const Tensor *a, const Tensor *b, binary_op_t op) {
    assert(a != NULL);
    assert(b != NULL);
    assert(op != NULL);
    assert(a->data != NULL || a->size == 0);
    assert(b->data != NULL || b->size == 0);
    if (a->size > 0) {
        assert((uintptr_t)a->data % CACHELINE_SIZE == 0 && "a->data is not properly aligned");
    }
    if (b->size > 0) {
        assert((uintptr_t)b->data % CACHELINE_SIZE == 0 && "b->data is not properly aligned");
    }

    uint64_t out_shape[MAX_NDIM];
    uint64_t out_ndim;
    if (!broadcast_shapes_mut(a->shape, a->ndim, b->shape, b->ndim, out_shape, &out_ndim)) {
        assert(false && "shapes cannot be broadcasted");
    }
    assert(out_ndim <= MAX_NDIM);
    Tensor *out_tensor = tensor_zeros(out_shape, out_ndim, a->requires_grad || b->requires_grad);

    // curr = current position in output tensor as multidim indices
    uint64_t *curr = (uint64_t *)calloc((size_t)out_ndim, sizeof(uint64_t));
    assert(curr != NULL && "calloc failed");

    // i = current position in output tensor as linear index
    for (uint64_t i = 0; i < out_tensor->size; i++) {
        // convert i to curr (mutates curr array)
        linear_to_multidim_mut(i, out_shape, out_ndim, curr);

        uint64_t offset_a = multidim_to_linear(curr, out_ndim, a->shape, a->ndim, a->strides);
        assert(offset_a < a->size && "offset_a out of bounds");

        uint64_t offset_b = multidim_to_linear(curr, out_ndim, b->shape, b->ndim, b->strides);
        assert(offset_b < b->size && "offset_b out of bounds");

        out_tensor->data[i] = op(a->data[offset_a], b->data[offset_b]);
    }

    free(curr);

    assert(out_tensor != NULL);
    assert(out_tensor->ndim == out_ndim);
    assert(out_tensor->data != NULL || out_tensor->size == 0);
    return out_tensor;
}

static float32_t op_add(float32_t a, float32_t b) { return a + b; }

// ------------------------------------------------------------------------------
// BEGIN AUTOGRAD STUFF
// ------------------------------------------------------------------------------

// this can be moved to arithmetic_backward.c
static void add_backward(Function *fn, const Tensor *grad_output) {
    assert(fn != NULL);
    assert(grad_output != NULL);
    assert(fn->num_inputs == 2);

    Tensor *a = fn->inputs[0];
    Tensor *b = fn->inputs[1];

    if (a != NULL && a->requires_grad) {
        Tensor *grad_a = tensor_add_backward_a(grad_output, a);
        accumulate_grad(a, grad_a);
    }

    if (b != NULL && b->requires_grad) {
        Tensor *grad_b = tensor_add_backward_b(grad_output, b);
        accumulate_grad(b, grad_b);
    }
}

Tensor *tensor_add(const Tensor *a, const Tensor *b) {
    Tensor *result = tensor_binary_op(a, b, op_add);

    // set up autograd if needed
    if (result->requires_grad) {
        Function *fn = arena_alloc_function();
        fn->apply = add_backward;
        fn->output = result;
        fn->num_inputs = 2;
        fn->inputs[0] = (Tensor *)a;
        fn->inputs[1] = (Tensor *)b;
        fn->pending_count = 0;
        fn->ctx = NULL;

        // increment pending_count for parents with grad_fn
        if (a->grad_fn != NULL) {
            a->grad_fn->pending_count++;
        }
        if (b->grad_fn != NULL) {
            b->grad_fn->pending_count++;
        }

        result->grad_fn = fn;
    }

    return result;
}

static float32_t op_sub(float32_t a, float32_t b) { return a - b; }
Tensor *tensor_sub(const Tensor *a, const Tensor *b) { return tensor_binary_op(a, b, op_sub); }

static float32_t op_mul(float32_t a, float32_t b) { return a * b; }

// ------------------------------------------------------------------------------
// END AUTOGRAD STUFF
// ------------------------------------------------------------------------------

// Backward function for mul: d(a*b)/da = b, d(a*b)/db = a
static void mul_backward(Function *fn, const Tensor *grad_output) {
    assert(fn != NULL);
    assert(grad_output != NULL);
    assert(fn->num_inputs == 2);

    Tensor *a = fn->inputs[0];
    Tensor *b = fn->inputs[1];

    if (a != NULL && a->requires_grad) {
        Tensor *grad_a = tensor_mul_backward_a(grad_output, a, b);
        accumulate_grad(a, grad_a);
    }

    if (b != NULL && b->requires_grad) {
        Tensor *grad_b = tensor_mul_backward_b(grad_output, a, b);
        accumulate_grad(b, grad_b);
    }
}

Tensor *tensor_mul(const Tensor *a, const Tensor *b) {
    Tensor *result = tensor_binary_op(a, b, op_mul);

    // Set up autograd if needed
    if (result->requires_grad) {
        Function *fn = arena_alloc_function();
        fn->apply = mul_backward;
        fn->output = result;
        fn->num_inputs = 2;
        fn->inputs[0] = (Tensor *)a;
        fn->inputs[1] = (Tensor *)b;
        fn->pending_count = 0;
        fn->ctx = NULL;

        // Increment pending_count for parents with grad_fn
        if (a->grad_fn != NULL) {
            a->grad_fn->pending_count++;
        }
        if (b->grad_fn != NULL) {
            b->grad_fn->pending_count++;
        }

        result->grad_fn = fn;
    }

    return result;
}

static float32_t op_div(float32_t a, float32_t b) { return a / b; }
Tensor *tensor_div(const Tensor *a, const Tensor *b) { return tensor_binary_op(a, b, op_div); }

Tensor *tensor_matmul(const Tensor *a, const Tensor *b) {
    assert(a != NULL);
    assert(b != NULL);
    assert(a->data != NULL);
    assert(b->data != NULL);
    assert((uintptr_t)a->data % CACHELINE_SIZE == 0 && "a->data is not properly aligned");
    assert((uintptr_t)b->data % CACHELINE_SIZE == 0 && "b->data is not properly aligned");
    assert(a->ndim >= 1 && b->ndim >= 1 && "matmul requires at least 1D tensors");
    assert(a->ndim == 2 && b->ndim == 2 && "only 2D matmul supported");
    assert(a->shape[1] == b->shape[0] && "inner dimensions must match");

    uint64_t M = a->shape[0];
    uint64_t K = a->shape[1];
    uint64_t N = b->shape[1];

    assert(M <= MAX_TENSOR_SIZE);
    assert(K <= MAX_TENSOR_SIZE);
    assert(N <= MAX_TENSOR_SIZE);

    const uint64_t out_shape[] = {M, N};
    Tensor *result = tensor_zeros(out_shape, 2, a->requires_grad || b->requires_grad);

    // naive algorithm
    for (uint64_t i = 0; i < M; i++) {
        for (uint64_t j = 0; j < N; j++) {
            float32_t sum = 0.0f;
            for (uint64_t k = 0; k < K; k++) {
                uint64_t a_offset = i * a->strides[0] + k * a->strides[1];
                uint64_t b_offset = k * b->strides[0] + j * b->strides[1];
                assert(a_offset < a->size && "a_offset out of bounds");
                assert(b_offset < b->size && "b_offset out of bounds");
                sum += a->data[a_offset] * b->data[b_offset];
            }
            uint64_t result_offset = i * result->strides[0] + j * result->strides[1];
            assert(result_offset < result->size && "result_offset out of bounds");
            result->data[result_offset] = sum;
        }
    }

    assert(result != NULL);
    assert(result->ndim == 2);
    assert(result->shape[0] == M);
    assert(result->shape[1] == N);
    return result;
}
