#include "tensor.h"
#include "autograd.h"
#include <assert.h>
#include <string.h>

Tensor *tensor_new(int64_t rows, int64_t cols, const double *data, bool requires_grad) {
    assert(rows > 0 && cols > 0 && "tensor dims must be positive");
    int64_t bytes = rows * cols * (int64_t)sizeof(double);
    Tensor *t = perm_alloc(sizeof(Tensor));
    *t = (Tensor){.data = perm_alloc(bytes), .grad = requires_grad ? perm_alloc(bytes) : NULL, .rows = rows, .cols = cols, .requires_grad = requires_grad};
    if (data != NULL) {
        memcpy(t->data, data, (size_t)bytes);
    }
    return t;
}
