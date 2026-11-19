#pragma once

#include "tensor.h"

enum {
    n_layer = 1,
    n_embed = 16,
    n_head = 4,
    block_size = 16,
    vocab_size = 27, // 26 letters plus a boundary token
    bos = vocab_size - 1,
    n_params = 3 + 6 * n_layer,
};

// linear weights are stored (in, out) so that y = x @ w
typedef struct {
    Tensor *attn_wq, *attn_wk, *attn_wv, *attn_wo, *mlp_fc1, *mlp_fc2;
} Layer;

// named access for the forward pass, flat access for the optimizer
typedef union {
    struct {
        Tensor *wte, *wpe, *lm_head;
        Layer layers[n_layer];
    };
    Tensor *all[n_params];
} Params;

Params params_new(void); // zero weights that require grad, in perm

Tensor *logits(const Params *p, const i64 *ids, i64 n);                                    // (n, vocab_size)
Tensor *loss(const Params *p, const i64 *ids, const i64 *targets, const f64 *mask, i64 n); // 1x1

// draws tokens from softmax(logits) until the boundary token, then writes them as letters
void sample(const Params *p, u64 seed, char name[block_size]);
