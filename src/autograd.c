#include "autograd.h"

static Node *tape = NULL;
static int64_t tape_len = 0;
static int64_t tape_cap = 0;

Tensor *tape_push(int64_t rows, int64_t cols, BackwardFn backward, int64_t n_in, Tensor *const *in, double aux, void *ctx) {
    CHECK(n_in <= MAX_INPUTS, "too many inputs");
    if (tape_len == tape_cap) {
        tape_cap = tape_cap ? 2 * tape_cap : 1024;
        tape = realloc(tape, (size_t)tape_cap * sizeof(Node));
        CHECK(tape != NULL, "realloc failed");
    }
    bool requires_grad = false;
    for (int64_t i = 0; i < n_in; i++) {
        requires_grad |= in[i]->requires_grad;
    }
    Node *node = &tape[tape_len++];
    *node = (Node){.out = tensor_new(rows, cols, NULL, requires_grad), .n_in = n_in, .backward = backward, .aux = aux, .ctx = ctx};
    for (int64_t i = 0; i < n_in; i++) {
        node->in[i] = in[i];
    }
    return node->out;
}

void tape_reset(void) {
    for (int64_t i = 0; i < tape_len; i++) {
        tensor_free(tape[i].out);
        free(tape[i].ctx);
    }
    tape_len = 0;
}

double *grad_of(Tensor *t) {
    if (!t->requires_grad) {
        return NULL;
    }
    if (t->grad == NULL) {
        t->grad = calloc((size_t)(t->rows * t->cols), sizeof(double));
        CHECK(t->grad != NULL, "calloc failed");
    }
    return t->grad;
}

void backward(Tensor *loss) {
    CHECK(loss->rows == 1 && loss->cols == 1, "loss must be 1x1");
    CHECK(loss->requires_grad, "loss does not depend on any param");
    CHECK(loss->grad == NULL, "backward called twice on one graph, call tape_reset in between");
    grad_of(loss)[0] = 1.0;
    for (int64_t i = tape_len - 1; i >= 0; i--) {
        if (tape[i].out->grad != NULL) {
            tape[i].backward(&tape[i]);
        }
    }
}
