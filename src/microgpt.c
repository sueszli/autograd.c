#include "microgpt.h"
#include "std/random.h"
#include <assert.h>
#include <math.h>

_Static_assert(sizeof(Params) == N_PARAMS * sizeof(Tensor *), "Params must be a flat array of tensors");

Params params_new(void) {
    Params p = {0};
    p.wte = tensor_new(SHAPE(VOCAB_SIZE, N_EMBED), NULL, true);
    p.wpe = tensor_new(SHAPE(BLOCK_SIZE, N_EMBED), NULL, true);
    p.lm_head = tensor_new(SHAPE(N_EMBED, VOCAB_SIZE), NULL, true);
    for (int64_t l = 0; l < N_LAYER; l++) {
        p.layers[l] = (Layer){
            .attn_wq = tensor_new(SHAPE(N_EMBED, N_EMBED), NULL, true),
            .attn_wk = tensor_new(SHAPE(N_EMBED, N_EMBED), NULL, true),
            .attn_wv = tensor_new(SHAPE(N_EMBED, N_EMBED), NULL, true),
            .attn_wo = tensor_new(SHAPE(N_EMBED, N_EMBED), NULL, true),
            .mlp_fc1 = tensor_new(SHAPE(N_EMBED, 4 * N_EMBED), NULL, true),
            .mlp_fc2 = tensor_new(SHAPE(4 * N_EMBED, N_EMBED), NULL, true),
        };
    }
    return p;
}

// x + attn(rms_norm(x)), heads attend causally and are merged before the output projection
static Tensor *attn(Tensor *x, const Layer *l) {
    const int64_t hd = N_EMBED / N_HEAD;
    Tensor *xn = rms_norm(x, 1e-5);
    Tensor *q = matmul(xn, l->attn_wq), *k = matmul(xn, l->attn_wk), *v = matmul(xn, l->attn_wv);
    Tensor *heads[N_HEAD];
    for (int64_t h = 0; h < N_HEAD; h++) {
        Tensor *scores = matmul(narrow(q, h * hd, hd), transpose(narrow(k, h * hd, hd)));
        Tensor *weights = softmax(masked_fill_upper(mul(scores, 1.0 / pow((double)hd, 0.5)), -1e9));
        heads[h] = matmul(weights, narrow(v, h * hd, hd));
    }
    return add(x, matmul(cat(heads, N_HEAD), l->attn_wo));
}

// x + fc2(relu(fc1(rms_norm(x))))
static Tensor *mlp(Tensor *x, const Layer *l) { return add(x, matmul(relu(matmul(rms_norm(x, 1e-5), l->mlp_fc1)), l->mlp_fc2)); }

Tensor *logits(const Params *p, const int64_t *ids, int64_t n) {
    assert(n > 0 && n <= BLOCK_SIZE && "logits: sequence length out of range");
    int64_t pos[BLOCK_SIZE];
    for (int64_t i = 0; i < n; i++) {
        pos[i] = i;
    }
    Tensor *x = rms_norm(add(embedding(ids, n, p->wte), embedding(pos, n, p->wpe)), 1e-5);
    for (int64_t l = 0; l < N_LAYER; l++) {
        x = mlp(attn(x, &p->layers[l]), &p->layers[l]);
    }
    return matmul(x, p->lm_head);
}

Tensor *loss(const Params *p, const int64_t *ids, const int64_t *targets, const double *mask, int64_t n) { return cross_entropy(logits(p, ids, n), targets, mask); }

// the boundary token both starts and ends a name
void sample(const Params *p, uint64_t seed, char name[BLOCK_SIZE]) {
    int64_t tokens[BLOCK_SIZE] = {BOS};
    for (int64_t n = 1; n < BLOCK_SIZE; n++) {
        int64_t next = random_choice(&seed, &softmax(logits(p, tokens, n))->data[(n - 1) * VOCAB_SIZE], VOCAB_SIZE);
        graph_reset();
        if (next == BOS) {
            return;
        }
        tokens[n] = next;
        name[n - 1] = (char)('a' + next);
    }
}
