#include "autograd.h"
#include <assert.h>
#include <string.h>

static Node tape[TAPE_CAP];
static Tensor outs[TAPE_CAP];
static int64_t tape_len = 0;
static alignas(16) unsigned char arena[ARENA_BYTES];
static int64_t arena_len = 0;

void *tape_alloc(int64_t bytes) {
    bytes = (bytes + 15) / 16 * 16;
    assert(arena_len + bytes <= ARENA_BYTES && "arena full, raise ARENA_BYTES");
    void *p = &arena[arena_len];
    arena_len += bytes;
    memset(p, 0, (size_t)bytes);
    return p;
}

Tensor *tape_push(int64_t rows, int64_t cols, BackwardFn backward, int64_t n_in, Tensor *const *in, double aux, void *ctx) {
    assert(n_in <= MAX_INPUTS && "too many inputs");
    assert(tape_len < TAPE_CAP && "tape full, raise TAPE_CAP");
    assert(rows > 0 && cols > 0 && "tensor dims must be positive");
    bool requires_grad = false;
    for (int64_t i = 0; i < n_in; i++) {
        requires_grad |= in[i]->requires_grad;
    }
    Tensor *out = &outs[tape_len];
    *out = (Tensor){.data = tape_alloc(rows * cols * (int64_t)sizeof(double)), .grad = NULL, .rows = rows, .cols = cols, .requires_grad = requires_grad};
    Node *node = &tape[tape_len++];
    *node = (Node){.out = out, .n_in = n_in, .backward = backward, .aux = aux, .ctx = ctx};
    for (int64_t i = 0; i < n_in; i++) {
        node->in[i] = in[i];
    }
    return out;
}

void tape_reset(void) {
    tape_len = 0;
    arena_len = 0;
}

double *grad_of(Tensor *t) {
    if (!t->requires_grad) {
        return NULL;
    }
    if (t->grad == NULL) {
        t->grad = tape_alloc(t->rows * t->cols * (int64_t)sizeof(double));
    }
    return t->grad;
}

void backward(Tensor *loss) {
    assert(loss->rows == 1 && loss->cols == 1 && "loss must be 1x1");
    assert(loss->requires_grad && "loss does not depend on any param");
    assert(loss->grad == NULL && "backward called twice on one graph, call tape_reset in between");
    grad_of(loss)[0] = 1.0;
    for (int64_t i = tape_len - 1; i >= 0; i--) {
        if (tape[i].out->grad != NULL) {
            tape[i].backward(&tape[i]);
        }
    }
}
