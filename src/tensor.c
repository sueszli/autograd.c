#include "tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

// helper to get size from shape
static int64_t calculate_size(const int64_t *shape, int64_t ndim) {
    int64_t size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

// helper to calculate strides
static void calculate_strides(const int64_t *shape, int64_t ndim, int64_t *strides) {
    int64_t stride = 1;
    for (int64_t i = ndim - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

// cppcheck-suppress staticFunction
Tensor *tensor_create(const float32_t *data, const int64_t *shape, int64_t ndim, bool requires_grad) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->ndim = ndim;

    // safety check for empty shape (scalar)
    if (ndim > 0 && shape != NULL) {
        t->shape = (int64_t *)malloc((size_t)ndim * sizeof(int64_t));
        if (!t->shape) {
            free(t);
            return NULL;
        }
        memcpy(t->shape, shape, (size_t)ndim * sizeof(int64_t));

        t->strides = (int64_t *)malloc((size_t)ndim * sizeof(int64_t));
        if (!t->strides) {
            free(t->shape);
            free(t);
            return NULL;
        }
        calculate_strides(t->shape, ndim, t->strides);
    } else {
        t->shape = NULL;
        t->strides = NULL;
        t->ndim = 0;
    }

    // determine size
    t->size = (ndim > 0) ? calculate_size(shape, ndim) : 1;

    // allocate data
    if (t->size > 0) {
        t->data = (float32_t *)malloc((size_t)t->size * sizeof(float32_t));
        if (!t->data) {
            if (t->strides) free(t->strides);
            if (t->shape) free(t->shape);
            free(t);
            return NULL;
        }
        if (data) {
            memcpy(t->data, data, (size_t)t->size * sizeof(float32_t));
        } else {
            memset(t->data, 0, (size_t)t->size * sizeof(float32_t));
        }
    } else {
        t->data = NULL;
    }

    t->requires_grad = requires_grad;
    t->grad = NULL;

    return t;
}

// cppcheck-suppress staticFunction
Tensor *tensor_zeros(const int64_t *shape, int64_t ndim, bool requires_grad) {
    return tensor_create(NULL, shape, ndim, requires_grad);
}

// cppcheck-suppress staticFunction
void tensor_free(Tensor *t) {
    if (!t) return;
    if (t->data) free(t->data);
    if (t->shape) free(t->shape);
    if (t->strides) free(t->strides);
    if (t->grad) tensor_free(t->grad);
    free(t);
}

// broadcasting logic
static bool broadcast_shapes(const int64_t *shape_a, int64_t ndim_a, const int64_t *shape_b, int64_t ndim_b, int64_t *out_shape, int64_t *out_ndim) {
    int64_t max_ndim = ndim_a > ndim_b ? ndim_a : ndim_b;
    *out_ndim = max_ndim;

    int64_t idx_a = ndim_a - 1;
    int64_t idx_b = ndim_b - 1;
    int64_t idx_out = max_ndim - 1;

    while (idx_out >= 0) {
        int64_t dim_a = (idx_a >= 0 && shape_a) ? shape_a[idx_a] : 1;
        int64_t dim_b = (idx_b >= 0 && shape_b) ? shape_b[idx_b] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) return false;

        out_shape[idx_out] = (dim_a > dim_b) ? dim_a : dim_b;

        idx_a--;
        idx_b--;
        idx_out--;
    }
    return true;
}

// 
// arithmetic
// 

typedef float32_t (*binary_op_t)(float32_t, float32_t);

static float32_t op_add(float32_t a, float32_t b) { return a + b; }
static float32_t op_sub(float32_t a, float32_t b) { return a - b; }
static float32_t op_mul(float32_t a, float32_t b) { return a * b; }
static float32_t op_div(float32_t a, float32_t b) { return a / b; }

static Tensor *tensor_binary_op(Tensor *a, Tensor *b, binary_op_t op) {
    int64_t out_shape[32];
    int64_t out_ndim;

    if (!broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, out_shape, &out_ndim)) {
        fprintf(stderr, "error: shapes cannot be broadcasted.\n");
        return NULL;
    }

    Tensor *result = tensor_zeros(out_shape, out_ndim, a->requires_grad || b->requires_grad);
    if (!result) return NULL;

    int64_t *indices = (int64_t *)calloc((size_t)out_ndim, sizeof(int64_t));
    if (out_ndim > 0 && !indices) {
        tensor_free(result);
        return NULL;
    }

    for (int64_t i = 0; i < result->size; i++) {
        if (out_ndim > 0) {
            int64_t temp = i;
            for (int64_t d = out_ndim - 1; d >= 0; d--) {
                indices[d] = temp % out_shape[d];
                temp /= out_shape[d];
            }
        }

        int64_t offset_a = 0;
        if (a->ndim > 0) {
            for (int64_t d = 0; d < a->ndim; d++) {
                int64_t result_dim_idx = d + (out_ndim - a->ndim);
                int64_t idx = (a->shape[d] == 1) ? 0 : indices[result_dim_idx];
                offset_a += idx * a->strides[d];
            }
        }

        int64_t offset_b = 0;
        if (b->ndim > 0) {
            for (int64_t d = 0; d < b->ndim; d++) {
                int64_t result_dim_idx = d + (out_ndim - b->ndim);
                int64_t idx = (b->shape[d] == 1) ? 0 : indices[result_dim_idx];
                offset_b += idx * b->strides[d];
            }
        }

        result->data[i] = op(a->data[offset_a], b->data[offset_b]);
    }

    if (indices) free(indices);
    return result;
}

Tensor *tensor_add(Tensor *a, Tensor *b) { return tensor_binary_op(a, b, op_add); }
Tensor *tensor_sub(Tensor *a, Tensor *b) { return tensor_binary_op(a, b, op_sub); }
Tensor *tensor_mul(Tensor *a, Tensor *b) { return tensor_binary_op(a, b, op_mul); }
Tensor *tensor_div(Tensor *a, Tensor *b) { return tensor_binary_op(a, b, op_div); }

Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    if (a->ndim < 1 || b->ndim < 1) {
        fprintf(stderr, "error: matmul requires at least 1d tensors.\n");
        return NULL;
    }

    if (a->ndim == 2 && b->ndim == 2) {
        if (a->shape[1] != b->shape[0]) {
            fprintf(stderr, "error: inner dimensions must match: %" PRId64 " != %" PRId64 "\n", a->shape[1], b->shape[0]);
            return NULL;
        }
        int64_t M = a->shape[0];
        int64_t K = a->shape[1];
        int64_t N = b->shape[1];

        const int64_t out_shape[] = {M, N};
        Tensor *result = tensor_zeros(out_shape, 2, a->requires_grad || b->requires_grad);
        if (!result) return NULL;

        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < N; j++) {
                float32_t sum = 0;
                for (int64_t k = 0; k < K; k++) {
                    sum += a->data[i * a->strides[0] + k * a->strides[1]] * b->data[k * b->strides[0] + j * b->strides[1]];
                }
                result->data[i * result->strides[0] + j * result->strides[1]] = sum;
            }
        }
        return result;
    }

    fprintf(stderr, "error: only 2d matmul supported in c implementation for now.\n");
    return NULL;
}

// 
// shape manipulation
// 

Tensor *tensor_reshape(const Tensor *t, const int64_t *new_shape, int64_t new_ndim) {
    int64_t new_size = 1;
    int64_t unknown_idx = -1;
    for (int64_t i = 0; i < new_ndim; i++) {
        if (new_shape[i] == -1) {
            if (unknown_idx != -1) {
                fprintf(stderr, "error: only one dimension can be -1.\n");
                return NULL;
            }
            unknown_idx = i;
        } else {
            new_size *= new_shape[i];
        }
    }

    if (unknown_idx != -1) {
        if (t->size % new_size != 0) {
            fprintf(stderr, "error: invalid shape.\n");
            return NULL;
        }
    } else {
        if (new_size != t->size) {
            fprintf(stderr, "error: total elements must match: %" PRId64 " != %" PRId64 "\n", t->size, new_size);
            return NULL;
        }
    }

    int64_t *resolved_shape = (int64_t *)malloc((size_t)new_ndim * sizeof(int64_t));
    if (!resolved_shape) return NULL;

    for (int64_t i = 0; i < new_ndim; i++) {
        resolved_shape[i] = new_shape[i];
    }
    if (unknown_idx != -1) {
        resolved_shape[unknown_idx] = t->size / new_size;
    }

    Tensor *result = tensor_create(t->data, resolved_shape, new_ndim, t->requires_grad);
    free(resolved_shape);
    return result;
}

Tensor *tensor_transpose(Tensor *t, int64_t dim0, int64_t dim1) {
    if (t->ndim < 2) {
        return tensor_create(t->data, t->shape, t->ndim, t->requires_grad);
    }

    int64_t *new_shape = (int64_t *)malloc((size_t)t->ndim * sizeof(int64_t));
    if (!new_shape) return NULL;

    memcpy(new_shape, t->shape, (size_t)t->ndim * sizeof(int64_t));
    int64_t temp = new_shape[dim0];
    new_shape[dim0] = new_shape[dim1];
    new_shape[dim1] = temp;

    Tensor *result = tensor_zeros(new_shape, t->ndim, t->requires_grad);
    free(new_shape);
    if (!result) return NULL;

    int64_t *indices = (int64_t *)calloc((size_t)t->ndim, sizeof(int64_t));
    if (!indices) {
        tensor_free(result);
        return NULL;
    }

    for (int64_t i = 0; i < result->size; i++) {
        int64_t temp_i = i;
        for (int64_t d = t->ndim - 1; d >= 0; d--) {
            indices[d] = temp_i % result->shape[d];
            temp_i /= result->shape[d];
        }

        int64_t offset = 0;
        for (int64_t d = 0; d < t->ndim; d++) {
            int64_t idx_val;
            if (d == dim0) idx_val = indices[dim1];
            else if (d == dim1) idx_val = indices[dim0];
            else idx_val = indices[d];

            offset += idx_val * t->strides[d];
        }

        result->data[i] = t->data[offset];
    }
    free(indices);

    return result;
}

// 
// reductions
// 

// cppcheck-suppress staticFunction
Tensor *tensor_sum(Tensor *t, int64_t axis, bool keepdims) {
    if (axis < 0) axis += t->ndim;
    if (axis < 0 || axis >= t->ndim) {
        // invalid axis
        return NULL;
    }

    int64_t *new_shape = NULL;
    if (t->ndim > 0) {
        new_shape = (int64_t *)malloc((size_t)t->ndim * sizeof(int64_t));
    }

    int64_t new_ndim = t->ndim;

    if (keepdims) {
        if (new_shape) {
            for (int64_t i = 0; i < t->ndim; i++) {
                new_shape[i] = (i == axis) ? 1 : t->shape[i];
            }
        }
    } else {
        new_ndim = t->ndim - 1;
        if (new_ndim > 0) {
            int64_t k = 0;
            for (int64_t i = 0; i < t->ndim; i++) {
                if (i != axis) new_shape[k++] = t->shape[i];
            }
        }
    }

    Tensor *result = tensor_zeros(new_shape, new_ndim, t->requires_grad);
    if (new_shape) free(new_shape);
    if (!result) return NULL;

    int64_t *indices = NULL;
    if (result->ndim > 0) {
        indices = (int64_t *)calloc((size_t)result->ndim, sizeof(int64_t));
        if (!indices) {
            tensor_free(result);
            return NULL;
        }
    }

    for (int64_t i = 0; i < result->size; i++) {
        if (result->ndim > 0) {
            int64_t temp_i = i;
            for (int64_t d = result->ndim - 1; d >= 0; d--) {
                indices[d] = temp_i % result->shape[d];
                temp_i /= result->shape[d];
            }
        }

        int64_t base_offset = 0;
        int64_t k = 0;
        for (int64_t d = 0; d < t->ndim; d++) {
            if (d == axis) {
            } else {
                int64_t idx_val = 0;
                if (indices) {
                    if (keepdims) idx_val = indices[d];
                    else idx_val = indices[k++];
                }
                base_offset += idx_val * t->strides[d];
            }
        }

        float32_t sum = 0;
        int64_t axis_dim = (t->shape) ? t->shape[axis] : 1;
        for (int64_t j = 0; j < axis_dim; j++) {
            sum += t->data[base_offset + j * t->strides[axis]];
        }
        result->data[i] = sum;
    }

    if (indices) free(indices);
    return result;
}

Tensor *tensor_mean(Tensor *t, int64_t axis, bool keepdims) {
    Tensor *sum = tensor_sum(t, axis, keepdims);
    if (!sum) return NULL;

    if (axis < 0) axis += t->ndim;
    int64_t n = (t->shape) ? t->shape[axis] : 1;

    for (int64_t i = 0; i < sum->size; i++) {
        sum->data[i] /= (float32_t)n;
    }

    return sum;
}

Tensor *tensor_max(Tensor *t, int64_t axis, bool keepdims) {
    if (axis < 0) axis += t->ndim;

    int64_t *new_shape = NULL;
    if (t->ndim > 0) {
        new_shape = (int64_t *)malloc((size_t)t->ndim * sizeof(int64_t));
    }
    int64_t new_ndim = t->ndim;

    if (keepdims) {
        if (new_shape) {
            for (int64_t i = 0; i < t->ndim; i++) {
                new_shape[i] = (i == axis) ? 1 : t->shape[i];
            }
        }
    } else {
        new_ndim = t->ndim - 1;
        if (new_ndim > 0) {
            int64_t k = 0;
            for (int64_t i = 0; i < t->ndim; i++) {
                if (i != axis) new_shape[k++] = t->shape[i];
            }
        }
    }

    Tensor *result = tensor_zeros(new_shape, new_ndim, t->requires_grad);
    if (new_shape) free(new_shape);
    if (!result) return NULL;

    int64_t *indices = NULL;
    if (result->ndim > 0) {
        indices = (int64_t *)calloc((size_t)result->ndim, sizeof(int64_t));
    }

    for (int64_t i = 0; i < result->size; i++) {
        if (indices) {
            int64_t temp_i = i;
            for (int64_t d = result->ndim - 1; d >= 0; d--) {
                indices[d] = temp_i % result->shape[d];
                temp_i /= result->shape[d];
            }
        }

        int64_t base_offset = 0;
        int64_t k = 0;
        for (int64_t d = 0; d < t->ndim; d++) {
            if (d == axis) {
            } else {
                int64_t idx_val = 0;
                if (indices) {
                    if (keepdims) idx_val = indices[d];
                    else idx_val = indices[k++];
                }
                base_offset += idx_val * t->strides[d];
            }
        }

        float32_t max_val = -INFINITY;
        int64_t axis_dim = (t->shape) ? t->shape[axis] : 1;
        if (axis_dim > 0) {
            max_val = t->data[base_offset];
            for (int64_t j = 1; j < axis_dim; j++) {
                float32_t val = t->data[base_offset + j * t->strides[axis]];
                if (val > max_val) max_val = val;
            }
        }
        result->data[i] = max_val;
    }

    if (indices) free(indices);
    return result;
}

// 
// utils
// 

static void tensor_print_recursive(Tensor *t, int64_t dim, int64_t offset, int indent) {
    if (dim == t->ndim) {
        printf("%f", t->data[offset]);
        return;
    }

    if (dim == t->ndim - 1) {
        printf("[");
        for (int64_t i = 0; i < t->shape[dim]; i++) {
            printf("%f", t->data[offset + i * t->strides[dim]]);
            if (i < t->shape[dim] - 1) {
                printf(", ");
            }
        }
        printf("]");
    } else {
        printf("[");
        for (int64_t i = 0; i < t->shape[dim]; i++) {
            if (i > 0) {
                for (int j = 0; j < indent; j++) printf(" ");
            }
            tensor_print_recursive(t, dim + 1, offset + i * t->strides[dim], indent + 1);

            if (i < t->shape[dim] - 1) {
                printf(",");
                int64_t newlines = t->ndim - dim - 1;
                for (int k = 0; k < newlines; k++) printf("\n");
            }
        }
        printf("]");
    }
}

void tensor_print(Tensor *t) {
    if (!t) {
        printf("Tensor(NULL)\n");
        return;
    }
    printf("Tensor(shape=[");
    if (t->shape) {
        for (int64_t i = 0; i < t->ndim; i++) {
            printf("%" PRId64 "%s", t->shape[i], i < t->ndim - 1 ? ", " : "");
        }
    }
    printf("], size=%" PRId64 ", requires_grad=%s)\n", t->size, t->requires_grad ? "true" : "false");

    if (t->data) {
        if (t->size <= 1000) {
            printf("Data: ");
            tensor_print_recursive(t, 0, 0, 6);
            printf("\n");
        } else {
            printf("Data: ... (size > 1000)\n");
        }
    }
}

Tensor *tensor_get(Tensor *t, const int64_t *indices) {
    int64_t offset = 0;
    if (t->ndim > 0 && indices) {
        for (int64_t i = 0; i < t->ndim; i++) {
            offset += indices[i] * t->strides[i];
        }
    }

    Tensor *res = tensor_create(&t->data[offset], NULL, 0, t->requires_grad);
    return res;
}
