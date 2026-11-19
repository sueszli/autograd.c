#pragma once

#include "tensor.h"

static constexpr i64 N_LAYER = 1;
static constexpr i64 N_EMBED = 16;
static constexpr i64 N_HEAD = 4;
static constexpr i64 BLOCK_SIZE = 16;
static constexpr i64 VOCAB_SIZE = 27; // 26 letters plus a boundary token
static constexpr i64 BOS = VOCAB_SIZE - 1;
static constexpr i64 N_PARAMS = 3 + 6 * N_LAYER;

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

typedef struct {
    // zero weights that require grad, in mem->perm
    Params (*params)(Mem *mem);

    // (n, VOCAB_SIZE)
    Tensor *(*logits)(const Params *p, const i64 *ids, i64 n);

    // scalar
    Tensor *(*loss)(const Params *p, const i64 *ids, const i64 *targets, const f64 *mask, i64 n);

    // draws tokens from softmax(logits) until the boundary token, then writes them as letters
    void (*sample)(const Params *p, u64 seed, char name[BLOCK_SIZE]);
} MicrogptNamespace;

extern const MicrogptNamespace microgpt;
