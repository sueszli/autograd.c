#pragma once

#include "autograd.h"

#define N_LAYER 1
#define N_EMBED 16
#define N_HEAD 4
#define BLOCK_SIZE 16
#define VOCAB_SIZE 27 // 26 letters plus a boundary token
#define BOS (VOCAB_SIZE - 1)
#define N_PARAMS (3 + 6 * N_LAYER)

// linear weights are stored (in, out) so that y = x @ w
typedef struct {
    Tensor *attn_wq, *attn_wk, *attn_wv, *attn_wo, *mlp_fc1, *mlp_fc2;
} Layer;

// named access for the forward pass, flat access for the optimizer
typedef union {
    struct {
        Tensor *wte, *wpe, *lm_head;
        Layer layers[N_LAYER];
    };
    Tensor *all[N_PARAMS];
} Params;

Params params_new(void); // zero weights that require grad, in perm

Tensor *logits(const Params *p, const int64_t *ids, int64_t n);                                           // (n, VOCAB_SIZE)
Tensor *loss(const Params *p, const int64_t *ids, const int64_t *targets, const double *mask, int64_t n); // 1x1

// draws tokens from softmax(logits) until the boundary token, then writes them as letters
void sample(const Params *p, uint64_t seed, char name[BLOCK_SIZE]);
