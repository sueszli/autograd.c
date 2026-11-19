#include "autograd.h"
#include "std/mem.h"
#include <assert.h>
#include <string.h>

static alignas(16) unsigned char perm_buf[512 * 1024];  // parity uses about 200 KB: two param sets, their grads and AdamW state
static alignas(16) unsigned char graph_buf[256 * 1024]; // one microGPT loss plus backward uses about 225 KB
static Arena perm = {.buf = perm_buf, .cap = sizeof(perm_buf)};
static Arena graph = {.buf = graph_buf, .cap = sizeof(graph_buf)};

void *perm_alloc(int64_t bytes) { return arena_alloc(&perm, (size_t)bytes); }
void perm_reset(void) { arena_reset(&perm); }
void *graph_alloc(int64_t bytes) { return arena_alloc(&graph, (size_t)bytes); }

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

// postorder puts each node after all the nodes it depends on
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
    assert(n_order == 0 && "backward called twice on one graph, call graph_reset in between");
    topo_sort(loss);
    grad_of(loss)[0] = 1.0;
    for (int64_t i = n_order - 1; i >= 0; i--) {
        order[i]->backward(order[i]);
    }
}

void graph_reset(void) {
    arena_reset(&graph);
    n_order = 0;
}
