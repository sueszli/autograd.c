#include "autograd.h"
#include <assert.h>

static constexpr i64 MAX_NODES = 128; // one microGPT loss uses 54

void *graph_alloc(Mem *mem, i64 bytes) { return arena.alloc(&mem->graph, (usize)bytes); }

Tensor *node_new(const i64 *shape, i64 ndim, BackwardFn backward, i64 n_in, Tensor *const *in, f64 aux, void *ctx) {
    // an op's node and its zeroed output, both in the graph of in[0]
    assert(n_in > 0 && n_in <= MAX_INPUTS && "op needs 1 to MAX_INPUTS inputs");
    Mem *mem = in[0]->mem;
    Node *node = graph_alloc(mem, sizeof(Node));
    *node = (Node){.n_in = n_in, .backward = backward, .aux = aux, .ctx = ctx};
    bool requires_grad = false;
    for (i64 i = 0; i < n_in; i++) {
        assert(in[i]->mem == mem && "op inputs must share one Mem");
        node->in[i] = in[i];
        requires_grad |= in[i]->requires_grad;
    }
    Tensor *out = graph_alloc(mem, sizeof(Tensor));
    set_shape(out, shape, ndim);
    out->data = graph_alloc(mem, out->size * (i64)sizeof(f64));
    out->node = node;
    out->mem = mem;
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
        t->grad = graph_alloc(t->mem, t->size * (i64)sizeof(f64));
    }
    return t->grad;
}

static void topo_sort(const Tensor *t, Node **order, i64 *n) {
    // postorder puts each node after all the nodes it depends on
    Node *node = t->node;
    if (!t->requires_grad || node == NULL || node->visited) {
        return;
    }
    node->visited = true;
    for (i64 i = 0; i < node->n_in; i++) {
        topo_sort(node->in[i], order, n);
    }
    assert(*n < MAX_NODES && "too many nodes, raise MAX_NODES");
    order[(*n)++] = node;
}

void autograd_backward(Tensor *loss) {
    // walks the sorted nodes from the loss down, the order lives in the graph too
    Mem *mem = loss->mem;
    assert(loss->size == 1 && "loss must have one element");
    assert(loss->requires_grad && "loss does not depend on any param");
    assert(!mem->backward_done && "backward called twice on one graph, call reset_graph in between");
    mem->backward_done = true;
    Node **order = graph_alloc(mem, MAX_NODES * (i64)sizeof(Node *));
    i64 n = 0;
    topo_sort(loss, order, &n);
    grad_of(loss)[0] = 1.0;
    for (i64 i = n - 1; i >= 0; i--) {
        order[i]->backward(order[i]);
    }
}

void autograd_reset(Mem *mem) {
    // frees every node, op output and grad of op outputs at once
    arena.reset(&mem->graph);
    mem->backward_done = false;
}
