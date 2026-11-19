#include "autograd.h"

static_assert(sizeof(Node) % 8 == 0 && sizeof(Tensor) % 8 == 0, "stack pushes round to 8 bytes");

static alignas(16) unsigned char nodes_buf[128 * sizeof(Node)];
static alignas(16) unsigned char mem_buf[256 * 1024];
Stack nodes = {.buf = nodes_buf, .cap = sizeof(nodes_buf)};
Stack mem = {.buf = mem_buf, .cap = sizeof(mem_buf)};

Tensor *node_push(int64_t rows, int64_t cols, BackwardFn backward, int64_t n_in, Tensor *const *in, double aux, void *ctx) {
    assert(n_in <= MAX_INPUTS && "too many inputs");
    assert(rows > 0 && cols > 0 && "tensor dims must be positive");
    bool requires_grad = false;
    for (int64_t i = 0; i < n_in; i++) {
        requires_grad |= in[i]->requires_grad;
    }
    Tensor *out = stack_push(&mem, sizeof(Tensor));
    *out = (Tensor){.data = stack_push(&mem, rows * cols * (int64_t)sizeof(double)), .rows = rows, .cols = cols, .requires_grad = requires_grad};
    Node *node = stack_push(&nodes, sizeof(Node));
    *node = (Node){.out = out, .n_in = n_in, .backward = backward, .aux = aux, .ctx = ctx};
    for (int64_t i = 0; i < n_in; i++) {
        node->in[i] = in[i];
    }
    return out;
}

void autograd_reset(void) {
    stack_reset(&nodes);
    stack_reset(&mem);
}

double *grad_of(Tensor *t) {
    if (!t->requires_grad) {
        return NULL;
    }
    if (t->grad == NULL) {
        t->grad = stack_push(&mem, t->rows * t->cols * (int64_t)sizeof(double));
    }
    return t->grad;
}

void backward(Tensor *loss) {
    assert(loss->rows == 1 && loss->cols == 1 && "loss must be 1x1");
    assert(loss->requires_grad && "loss does not depend on any param");
    assert(loss->grad == NULL && "backward called twice on one graph, call autograd_reset in between");
    grad_of(loss)[0] = 1.0;
    const Node *base = (const Node *)nodes.buf;
    for (int64_t i = nodes.len / (int64_t)sizeof(Node) - 1; i >= 0; i--) {
        if (base[i].out->grad != NULL) {
            base[i].backward(&base[i]);
        }
    }
}
