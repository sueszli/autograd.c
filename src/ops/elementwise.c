#include "autograd.h"
#include "ops.h"
#include <assert.h>

static void add_backward(const Node *node);
static void mul_backward(const Node *node);
static void relu_backward(const Node *node);

Tensor *add(Tensor *a, Tensor *b) {
    assert(a->ndim == b->ndim && a->size == b->size && "add: shape mismatch");
    Tensor *out = node_new(a->shape, a->ndim, add_backward, 2, (Tensor *[]){a, b}, 0, NULL);
    for (i64 i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return out;
}

static void add_backward(const Node *node) {
    // both inputs get the output grad
    const Tensor *out = node->out;
    for (i64 i = 0; i < node->n_in; i++) {
        f64 *g = grad_of(node->in[i]);
        for (i64 j = 0; g != NULL && j < out->size; j++) {
            g[j] += out->grad[j];
        }
    }
}

Tensor *mul(Tensor *a, f64 s) {
    Tensor *out = node_new(a->shape, a->ndim, mul_backward, 1, (Tensor *[]){a}, s, NULL);
    for (i64 i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] * s;
    }
    return out;
}

static void mul_backward(const Node *node) {
    // d(a * s) / da = s
    const Tensor *out = node->out;
    f64 *g = grad_of(node->in[0]);
    for (i64 i = 0; g != NULL && i < out->size; i++) {
        g[i] += node->aux * out->grad[i];
    }
}

Tensor *relu(Tensor *a) {
    Tensor *out = node_new(a->shape, a->ndim, relu_backward, 1, (Tensor *[]){a}, 0, NULL);
    for (i64 i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] > 0.0 ? a->data[i] : 0.0;
    }
    return out;
}

static void relu_backward(const Node *node) {
    // the grad passes where the input was positive
    const Tensor *a = node->in[0], *out = node->out;
    f64 *g = grad_of(node->in[0]);
    for (i64 i = 0; g != NULL && i < out->size; i++) {
        g[i] += a->data[i] > 0.0 ? out->grad[i] : 0.0;
    }
}
