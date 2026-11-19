#include "tensor.h"
#include "alloc.h"

Tensor *tensor_new(int64_t rows, int64_t cols, const double *data, bool requires_grad) {
    assert(rows > 0 && cols > 0 && "tensor dims must be positive");
    int64_t bytes = rows * cols * (int64_t)sizeof(double);
    Tensor *t = fba_alloc(&perm_mem, sizeof(Tensor));
    *t = (Tensor){.data = fba_alloc(&perm_mem, bytes), .grad = requires_grad ? fba_alloc(&perm_mem, bytes) : NULL, .rows = rows, .cols = cols, .requires_grad = requires_grad};
    if (data != NULL) {
        memcpy(t->data, data, (size_t)bytes);
    }
    return t;
}
