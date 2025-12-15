#include "ops/reshapes_backward.h"
#include "ops/reshapes.h"
#include "tensor.h"
#include <assert.h>
#include <stddef.h>

Tensor *tensor_reshape_backward(const Tensor *grad_output, const Tensor *input) {
    assert(grad_output != NULL);
    assert(input != NULL);
    return tensor_reshape(grad_output, (const int64_t *)input->shape, input->ndim);
}

Tensor *tensor_transpose_backward(const Tensor *grad_output, uint64_t dim0, uint64_t dim1) {
    assert(grad_output != NULL);
    return tensor_transpose(grad_output, dim0, dim1);
}
