#include "autograd.h"
#include "microgpt.h"
#include "optim.h"
#include "std/assert.h"
#include "std/json.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

enum { num_steps = 1000 };

static const char *param_names[n_params] = {"wte", "wpe", "lm_head", "layer0.attn_wq", "layer0.attn_wk", "layer0.attn_wv", "layer0.attn_wo", "layer0.mlp_fc1", "layer0.mlp_fc2"};

// the reference stores linear weights as (out, in), wte and wpe are already (rows, cols)
static void load_params(const cJSON *j, Params *p) {
    for (i64 i = 0; i < n_params; i++) {
        json_matrix(json_get(j, param_names[i]), p->all[i]->data, p->all[i]->shape[0], p->all[i]->shape[1], i >= 2);
    }
}

// decoded once before training, so the training loop does no json work
static i64 inputs[num_steps][block_size], targets[num_steps][block_size];
static f64 masks[num_steps][block_size];

// json numbers are doubles, token ids are ints
static void load_ids(const cJSON *j, i64 ids[num_steps][block_size]) {
    static f64 raw[num_steps][block_size];
    json_matrix(j, &raw[0][0], num_steps, block_size, false);
    for (i64 s = 0; s < num_steps; s++) {
        for (i64 i = 0; i < block_size; i++) {
            ids[s][i] = (i64)raw[s][i];
        }
    }
}

// every final weight within 1e-11 of the reference
static void check_parity(const Params *p, const Params *ref) {
    f64 worst = 0.0;
    for (i64 i = 0; i < n_params; i++) {
        worst = fmax(worst, assert_allclose(p->all[i]->data, ref->all[i]->data, p->all[i]->size, 0.0, 1e-11, param_names[i]));
    }
    printf("passed parity check (max abs diff %.2g)\n", worst);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0); // unbuffered, so stdio never mallocs a buffer
    const cJSON *j = json_load("parity.json");
    assert((i64)json_get(j, "bos")->valuedouble == bos);
    load_ids(json_get(j, "inputs"), inputs);
    load_ids(json_get(j, "targets"), targets);
    json_matrix(json_get(j, "masks"), &masks[0][0], num_steps, block_size, false);
    Params p = params_new(), ref = params_new();
    load_params(json_get(j, "init_weights"), &p);
    load_params(json_get(j, "final_weights"), &ref);

    AdamW opt = adamw_new(p.all, n_params);
    clock_t start = clock();
    for (i64 step = 0; step < num_steps; step++) {
        backward(loss(&p, inputs[step], targets[step], masks[step], block_size));
        graph_reset();
        adamw_step(&opt, step + 1, num_steps);
    }
    printf("trained %d steps in %.3fs\n", num_steps, (f64)(clock() - start) / CLOCKS_PER_SEC);

    check_parity(&p, &ref);

    char name[block_size] = {0};
    sample(&p, (u64)time(NULL), name);
    printf("inference (random sampled name): %s\n", name);

    return 0;
}
