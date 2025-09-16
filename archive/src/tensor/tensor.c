#include "tensor.h"
#include "../utils/types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

u64 tensor_size(const tensor_t *t) {
    if (t == NULL)
        return 0;
    u64 size = 1;
    for (i32 i = 0; i < t->ndim; i++) {
        size *= (u64)t->shape[i];
    }
    return size;
}

tensor_t *tensor_create(f32 *data, i32 *shape, i32 ndim, bool requires_grad) {
    tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
    t->shape = (i32 *)malloc((u64)ndim * sizeof(i32));
    memcpy(t->shape, shape, (u64)ndim * sizeof(i32));
    t->ndim = ndim;

    u64 size = 1;
    for (i32 i = 0; i < ndim; i++) {
        size *= (u64)shape[i];
    }

    t->data = (f32 *)malloc(size * sizeof(f32));
    if (data != NULL) {
        memcpy(t->data, data, size * sizeof(f32));
    }

    t->requires_grad = requires_grad;
    t->grad = NULL;
    t->grad_fn = NULL;
    t->ctx = NULL;
    t->ctx_size = 0;

    return t;
}

void tensor_destroy(tensor_t *t) {
    if (t == NULL)
        return;
    free(t->data);
    free(t->shape);
    if (t->grad) {
        tensor_destroy(t->grad);
    }
    if (t->ctx) {
        free(t->ctx);
    }
    free(t);
}

// print a tensor (simple implementation)
void tensor_print(const tensor_t *t) {
    printf("tensor_t (shape: [");
    for (i32 i = 0; i < t->ndim; i++) {
        printf("%d", t->shape[i]);
        if (i < t->ndim - 1)
            printf(", ");
    }
    printf("], requires_grad: %s)\n", t->requires_grad ? "true" : "false");

    u64 size = tensor_size(t);
    for (u64 i = 0; i < size; i++) {
        printf("%f ", t->data[i]);
    }
    printf("\n");
}

void tensor_zero_grad(tensor_t *t) {
    if (t->grad) {
        memset(t->grad->data, 0, tensor_size(t->grad) * sizeof(f32));
    }
}

tensor_t *tensor_transpose(tensor_t *a) {
    if (a->ndim != 2)
        return NULL;
    i32 new_shape[] = {a->shape[1], a->shape[0]};
    f32 *new_data = malloc(tensor_size(a) * sizeof(f32));

    for (i32 i = 0; i < a->shape[0]; i++) {
        for (i32 j = 0; j < a->shape[1]; j++) {
            new_data[j * a->shape[0] + i] = a->data[i * a->shape[1] + j];
        }
    }

    tensor_t *result = tensor_create(new_data, new_shape, 2, a->requires_grad);
    free(new_data);
    result->requires_grad = false; // transpose backward is not implemented for simplicity

    return result;
}
