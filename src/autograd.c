#include "autograd.h"
#include <assert.h>
#include <string.h>

static alignas(16) unsigned char perm_buf[512 * 1024];  // parity uses about 200 KB: two param sets, their grads and AdamW state
static alignas(16) unsigned char graph_buf[256 * 1024]; // one microGPT loss plus backward uses about 225 KB
FixedBufferAllocator perm_mem = {.buffer = perm_buf, .buffer_len = sizeof(perm_buf)};
FixedBufferAllocator graph_mem = {.buffer = graph_buf, .buffer_len = sizeof(graph_buf)};

// zig's allocator.alloc followed by @memset(.., 0)
static void *alloc_zeroed(FixedBufferAllocator *a, int64_t bytes) {
    assert(bytes >= 0);
    void *p = fixed_buffer_allocator_alloc(a, (size_t)bytes, alignof(double));
    assert(p != NULL && "out of memory, raise the buffer size in autograd.c");
    return memset(p, 0, (size_t)bytes);
}

void *perm_alloc(int64_t bytes) { return alloc_zeroed(&perm_mem, bytes); }
void *graph_alloc(int64_t bytes) { return alloc_zeroed(&graph_mem, bytes); }

Tensor *node_new(int64_t rows, int64_t cols, BackwardFn backward, int64_t n_in, Tensor *const *in, double aux, void *ctx) {
    assert(n_in <= MAX_INPUTS && "too many inputs");
    assert(rows > 0 && cols > 0 && "tensor dims must be positive");
    Node *node = graph_alloc(sizeof(Node));
    *node = (Node){.n_in = n_in, .backward = backward, .aux = aux, .ctx = ctx};
    bool requires_grad = false;
    for (int64_t i = 0; i < n_in; i++) {
        node->in[i] = in[i];
        requires_grad |= in[i]->requires_grad;
    }
    node->out = graph_alloc(sizeof(Tensor));
    *node->out = (Tensor){.data = graph_alloc(rows * cols * (int64_t)sizeof(double)), .node = node, .rows = rows, .cols = cols, .requires_grad = requires_grad};
    return node->out;
}

void graph_reset(void) { fixed_buffer_allocator_reset(&graph_mem); }

double *grad_of(Tensor *t) {
    if (!t->requires_grad) {
        return NULL;
    }
    if (t->grad == NULL) {
        t->grad = graph_alloc(t->rows * t->cols * (int64_t)sizeof(double));
    }
    return t->grad;
}

static Node *order[MAX_NODES];
static int64_t n_order;

// postorder: a node lands in order only after all nodes it depends on
static void topo_sort(const Tensor *t) {
    Node *node = t->node;
    if (!t->requires_grad || node == NULL || node->visited) {
        return;
    }
    node->visited = true;
    for (int64_t i = 0; i < node->n_in; i++) {
        topo_sort(node->in[i]);
    }
    assert(n_order < MAX_NODES && "too many nodes, raise MAX_NODES");
    order[n_order++] = node;
}

void backward(Tensor *loss) {
    assert(loss->rows == 1 && loss->cols == 1 && "loss must be 1x1");
    assert(loss->requires_grad && "loss does not depend on any param");
    assert(loss->grad == NULL && "backward called twice on one graph, call graph_reset in between");
    n_order = 0;
    topo_sort(loss);
    grad_of(loss)[0] = 1.0;
    for (int64_t i = n_order - 1; i >= 0; i--) {
        order[i]->backward(order[i]);
    }
}
