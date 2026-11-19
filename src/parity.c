#include "autograd.h"
#include "microgpt.h"
#include "optim.h"
#include "std/assert.h"
#include "std/json.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define NUM_STEPS 1000

static const char *NAMES[N_PARAMS] = {"wte", "wpe", "lm_head", "layer0.attn_wq", "layer0.attn_wk", "layer0.attn_wv", "layer0.attn_wo", "layer0.mlp_fc1", "layer0.mlp_fc2"};

// the reference stores linear weights as (out, in), wte and wpe are already (rows, cols)
static void load_params(const cJSON *j, Params *p) {
    for (int64_t i = 0; i < N_PARAMS; i++) {
        json_matrix(json_get(j, NAMES[i]), p->all[i]->data, p->all[i]->rows, p->all[i]->cols, i >= 2);
    }
}

// decoded once before training, so the training loop does no json work
static int64_t inputs[NUM_STEPS][BLOCK_SIZE], targets[NUM_STEPS][BLOCK_SIZE];
static double masks[NUM_STEPS][BLOCK_SIZE];

// json numbers are doubles, token ids are ints
static void load_ids(const cJSON *j, int64_t ids[NUM_STEPS][BLOCK_SIZE]) {
    static double raw[NUM_STEPS][BLOCK_SIZE];
    json_matrix(j, &raw[0][0], NUM_STEPS, BLOCK_SIZE, false);
    for (int64_t s = 0; s < NUM_STEPS; s++) {
        for (int64_t i = 0; i < BLOCK_SIZE; i++) {
            ids[s][i] = (int64_t)raw[s][i];
        }
    }
}

// every final weight within 1e-11 of the reference
static void check_parity(const Params *p, const Params *ref) {
    double worst = 0.0;
    for (int64_t i = 0; i < N_PARAMS; i++) {
        worst = fmax(worst, assert_allclose(p->all[i]->data, ref->all[i]->data, p->all[i]->rows * p->all[i]->cols, 0.0, 1e-11, NAMES[i]));
    }
    printf("passed parity check (max abs diff %.2g)\n", worst);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0); // unbuffered, so stdio never mallocs a buffer
    const cJSON *j = json_load("parity.json");
    assert(json_get(j, "bos")->valuedouble == BOS);
    load_ids(json_get(j, "inputs"), inputs);
    load_ids(json_get(j, "targets"), targets);
    json_matrix(json_get(j, "masks"), &masks[0][0], NUM_STEPS, BLOCK_SIZE, false);
    Params p = params_new(), ref = params_new();
    load_params(json_get(j, "init_weights"), &p);
    load_params(json_get(j, "final_weights"), &ref);

    AdamW opt = adamw_new(p.all, N_PARAMS);
    clock_t start = clock();
    for (int64_t step = 0; step < NUM_STEPS; step++) {
        backward(loss(&p, inputs[step], targets[step], masks[step], BLOCK_SIZE));
        graph_reset();
        adamw_step(&opt, step + 1, NUM_STEPS);
    }
    printf("trained %d steps in %.3fs\n", NUM_STEPS, (double)(clock() - start) / CLOCKS_PER_SEC);

    check_parity(&p, &ref);

    char name[BLOCK_SIZE] = {0};
    sample(&p, (uint64_t)time(NULL), name);
    printf("inference (random sampled name): %s\n", name);

    return 0;
}
