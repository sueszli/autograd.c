#include "autograd.h"
#include "std/mem.h"
#include <assert.h>

static constexpr i64 MAX_NODES = 128; // one microGPT loss uses 54

static alignas(16) u8 graph_buf[256 * 1024]; // one microGPT loss plus backward uses about 225 KB
static Arena graph = {.buf = graph_buf, .cap = sizeof(graph_buf)};

void *graph_alloc(i64 bytes) { return arena_alloc(&graph, (usize)bytes); }

Tensor *node_new(const i64 *shape, i64 ndim, BackwardFn backward, i64 n_in, Tensor *const *in, f64 aux, void *ctx) {
    // an op's node and its zeroed output, both in graph
    assert(n_in <= MAX_INPUTS && "too many inputs");
    Node *node = graph_alloc(sizeof(Node));
    *node = (Node){.n_in = n_in, .backward = backward, .aux = aux, .ctx = ctx};
    bool requires_grad = false;
    for (i64 i = 0; i < n_in; i++) {
        node->in[i] = in[i];
        requires_grad |= in[i]->requires_grad;
    }
    Tensor *out = graph_alloc(sizeof(Tensor));
    tensor_set_shape(out, shape, ndim);
    out->data = graph_alloc(out->size * (i64)sizeof(f64));
    out->node = node;
    out->requires_grad = requires_grad;
    node->out = out;
    return out;
}

f64 *grad_of(Tensor *t) {
    // NULL if t does not require grad, op outputs get a buffer on first use
    if (!t->requires_grad) {
        return NULL;
    }
    if (t->grad == NULL) {
        t->grad = graph_alloc(t->size * (i64)sizeof(f64));
    }
    return t->grad;
}

static Node *order[MAX_NODES];
static i64 n_order;

static void topo_sort(const Tensor *t) {
    // postorder puts each node after all the nodes it depends on
    Node *node = t->node;
    if (!t->requires_grad || node == NULL || node->visited) {
        return;
    }
    node->visited = true;
    for (i64 i = 0; i < node->n_in; i++) {
        topo_sort(node->in[i]);
    }
    assert(n_order < MAX_NODES && "too many nodes, raise max_nodes");
    order[n_order++] = node;
}

void backward(Tensor *loss) {
    // walks the sorted nodes from the loss down
    assert(loss->size == 1 && "loss must have one element");
    assert(loss->requires_grad && "loss does not depend on any param");
    assert(n_order == 0 && "backward called twice on one graph, call graph_reset in between");
    topo_sort(loss);
    grad_of(loss)[0] = 1.0;
    for (i64 i = n_order - 1; i >= 0; i--) {
        order[i]->backward(order[i]);
    }
}

void graph_reset(void) {
    // frees every node, op output and grad of op outputs at once
    arena_reset(&graph);
    n_order = 0;
}
