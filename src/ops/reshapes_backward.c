#include "ops/reshapes_backward.h"
#include "ops/reshapes.h"
#include "tensor.h"
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>

//
// reshape
//

Tensor *tensor_reshape_backward(const Tensor *grad_output, const Tensor *input) {
    assert(grad_output != NULL);
    assert(input != NULL);
    return tensor_reshape(grad_output, (const int64_t *)input->shape, input->ndim);
}

void reshape_backward(Function *fn, const Tensor *grad_output) {
    assert(fn != NULL);
    assert(grad_output != NULL);
    assert(fn->num_inputs == 1);
    assert(fn->ctx != NULL && "reshape_backward requires context");

    Tensor *t = fn->inputs[0];
    const ReshapeContext *ctx = (ReshapeContext *)fn->ctx;

    if (t != NULL && t->requires_grad) {
        // Create temporary tensor with original shape for backward function
        Tensor temp_input;
        temp_input.shape = (uint64_t *)ctx->shape;
        temp_input.ndim = ctx->ndim;

        Tensor *grad_t = tensor_reshape_backward(grad_output, &temp_input);
        accumulate_grad(t, grad_t);
    }

    free(fn->ctx);
    fn->ctx = NULL;
}

//
// transpose
//

Tensor *tensor_transpose_backward(const Tensor *grad_output, uint64_t dim0, uint64_t dim1) {
    assert(grad_output != NULL);
    return tensor_transpose(grad_output, dim0, dim1);
}

void transpose_backward(Function *fn, const Tensor *grad_output) {
    assert(fn != NULL);
    assert(grad_output != NULL);
    assert(fn->num_inputs == 1);
    assert(fn->ctx != NULL && "transpose_backward requires context");

    Tensor *t = fn->inputs[0];
    const TransposeContext *ctx = (TransposeContext *)fn->ctx;

    if (t != NULL && t->requires_grad) {
        Tensor *grad_t = tensor_transpose_backward(grad_output, ctx->dim0, ctx->dim1);
        accumulate_grad(t, grad_t);
    }

    free(fn->ctx);
    fn->ctx = NULL;
}
