#include "tensor.h"
#include <assert.h>
#include <string.h>

Tensor *tensor_new(int64_t rows, int64_t cols, const double *data, bool requires_grad) {
    assert(rows > 0 && cols > 0 && "tensor dims must be positive");
    Tensor *t = malloc(sizeof(Tensor));
    assert(t != NULL && "malloc failed");
    size_t n = (size_t)(rows * cols);
    *t = (Tensor){.data = calloc(n, sizeof(double)), .grad = requires_grad ? calloc(n, sizeof(double)) : NULL, .rows = rows, .cols = cols, .requires_grad = requires_grad};
    assert(t->data != NULL && (t->grad != NULL || !requires_grad) && "calloc failed");
    if (data != NULL) {
        memcpy(t->data, data, n * sizeof(double));
    }
    return t;
}

void tensor_free(Tensor *t) {
    if (t == NULL) {
        return;
    }
    free(t->data);
    free(t->grad);
    free(t);
}
