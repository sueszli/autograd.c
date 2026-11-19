#pragma once

#include "std/mem.h"
#include "std/types.h"

static constexpr i64 MAX_NDIM = 4;

//
// n-d row-major f64 tensors with reverse-mode autograd, named after pytorch.
// https://pytorch.org/docs/stable/tensors.html
//
// row-wise ops act on the last dim, and leading dims are a batch.
// every tensor points to the Mem it lives in.
//

typedef struct {
    Arena perm;  // params and optimizer state
    Arena graph; // op outputs, reset after each step
    bool backward_done;
} Mem;

typedef struct Tensor {
    f64 *data;
    f64 *grad;
    struct Node *node; // the op that made it, NULL for params
    Mem *mem;
    i64 shape[MAX_NDIM];
    i64 ndim; // 0 for a scalar
    i64 size;
    bool requires_grad;
} Tensor;

// shape(2, 3) expands to a shape literal and its length
#define shape(...) (const i64[]){__VA_ARGS__}, (i64)(sizeof((i64[]){__VA_ARGS__}) / sizeof(i64))

typedef struct {
    // data NULL gives zeros
    Tensor *(*new)(Mem *mem, const i64 *shape, i64 ndim, const f64 *data, bool requires_grad);

    void (*backward)(Tensor *loss); // param grads accumulate
    void (*reset_graph)(Mem *mem);  // call after each backward

    // elementwise
    Tensor *(*add)(Tensor *a, Tensor *b); // a + b, same shape
    Tensor *(*mul)(Tensor *a, f64 s);     // a * s
    Tensor *(*relu)(Tensor *a);           // max(a, 0)

    // matrix
    Tensor *(*matmul)(Tensor *a, Tensor *b); // (..., n, k) @ (k, m) -> (..., n, m)
    Tensor *(*transpose)(Tensor *a);         // swaps the last two dims

    // indexing
    Tensor *(*embedding)(const i64 *ids, i64 n, Tensor *table); // rows of a 2-d table picked by ids, (n, d)
    Tensor *(*masked_fill_upper)(Tensor *a, f64 value);         // entries above the diagonal set to value, last two dims square
    Tensor *(*narrow)(Tensor *a, i64 start, i64 len);           // last dim [start, start + len)
    Tensor *(*cat)(Tensor *const *parts, i64 n);                // parts joined along the last dim

    // normalization and loss
    Tensor *(*rms_norm)(Tensor *a, f64 eps);                                       // each row divided by its root mean square
    Tensor *(*softmax)(Tensor *a);                                                 // softmax over the last dim
    Tensor *(*cross_entropy)(Tensor *logits, const i64 *targets, const f64 *mask); // scalar, mean over rows with mask 1
} TensorNamespace;

extern const TensorNamespace tensor;
