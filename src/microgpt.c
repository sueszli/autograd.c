#include "microgpt.h"
#include "ops.h"
#include <assert.h>
#include <math.h>

_Static_assert(sizeof(Params) == N_PARAMS * sizeof(Tensor *), "Params must be a flat array of tensors");

Params params_new(void) {
    Params p = {0};
    p.wte = tensor_new(VOCAB_SIZE, N_EMBED, NULL, true);
    p.wpe = tensor_new(BLOCK_SIZE, N_EMBED, NULL, true);
    p.lm_head = tensor_new(N_EMBED, VOCAB_SIZE, NULL, true);
    for (int64_t l = 0; l < N_LAYER; l++) {
        p.layers[l] = (Layer){
            .attn_wq = tensor_new(N_EMBED, N_EMBED, NULL, true),
            .attn_wk = tensor_new(N_EMBED, N_EMBED, NULL, true),
            .attn_wv = tensor_new(N_EMBED, N_EMBED, NULL, true),
            .attn_wo = tensor_new(N_EMBED, N_EMBED, NULL, true),
            .mlp_fc1 = tensor_new(N_EMBED, 4 * N_EMBED, NULL, true),
            .mlp_fc2 = tensor_new(4 * N_EMBED, N_EMBED, NULL, true),
        };
    }
    return p;
}

// x + attn(rmsnorm(x)), heads attend causally and are merged before the output projection
static Tensor *attn(Tensor *x, const Layer *l) {
    const int64_t hd = N_EMBED / N_HEAD;
    Tensor *xn = rmsnorm(x, 1e-5);
    Tensor *q = matmul(xn, l->attn_wq), *k = matmul(xn, l->attn_wk), *v = matmul(xn, l->attn_wv);
    Tensor *heads[N_HEAD];
    for (int64_t h = 0; h < N_HEAD; h++) {
        Tensor *scores = matmul(slice_cols(q, h * hd, hd), transpose(slice_cols(k, h * hd, hd)));
        Tensor *weights = softmax_rows(causal_mask(scale(scores, 1.0 / pow((double)hd, 0.5)), -1e9));
        heads[h] = matmul(weights, slice_cols(v, h * hd, hd));
    }
    return add(x, matmul(concat_cols(heads, N_HEAD), l->attn_wo));
}

// x + fc2(relu(fc1(rmsnorm(x))))
static Tensor *mlp(Tensor *x, const Layer *l) { return add(x, matmul(relu(matmul(rmsnorm(x, 1e-5), l->mlp_fc1)), l->mlp_fc2)); }

Tensor *logits(const Params *p, const int64_t *ids, int64_t n) {
    assert(n > 0 && n <= BLOCK_SIZE && "logits: sequence length out of range");
    int64_t pos[BLOCK_SIZE];
    for (int64_t i = 0; i < n; i++) {
        pos[i] = i;
    }
    Tensor *x = rmsnorm(add(gather(p->wte, ids, n), gather(p->wpe, pos, n)), 1e-5);
    for (int64_t l = 0; l < N_LAYER; l++) {
        x = mlp(attn(x, &p->layers[l]), &p->layers[l]);
    }
    return matmul(x, p->lm_head);
}

Tensor *loss(const Params *p, const int64_t *ids, const int64_t *targets, const double *mask, int64_t n) { return masked_cross_entropy(logits(p, ids, n), targets, mask); }

// the boundary token both starts and ends a name
void sample(const Params *p, uint64_t seed, char name[BLOCK_SIZE]) {
    int64_t tokens[BLOCK_SIZE] = {BOS};
    for (int64_t n = 1; n < BLOCK_SIZE; n++) {
        const double *probs = &softmax_rows(logits(p, tokens, n))->data[(n - 1) * VOCAB_SIZE];
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; // 64-bit lcg, as in autograd.lean
        double u = (double)(seed >> 11) / 9007199254740992.0;          // uniform in [0, 1)
        int64_t next = VOCAB_SIZE - 1;
        for (int64_t c = 0; c < VOCAB_SIZE; c++) {
            u -= probs[c];
            if (u < 0.0) {
                next = c;
                break;
            }
        }
        graph_reset();
        if (next == BOS) {
            return;
        }
        tokens[n] = next;
        name[n - 1] = (char)('a' + next);
    }
}
