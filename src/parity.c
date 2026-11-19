#include "microgpt.h"
#include "optim.h"
#include "std/assert.h"
#include "std/json.h"
#include "std/mem.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

static constexpr i64 NUM_STEPS = 1000;

static const char *PARAM_NAMES[N_PARAMS] = {"wte", "wpe", "lm_head", "layer0.attn_wq", "layer0.attn_wk", "layer0.attn_wv", "layer0.attn_wo", "layer0.mlp_fc1", "layer0.mlp_fc2"};

// the reference stores linear weights as (out, in), wte and wpe are already (rows, cols)
static void load_params(const cJSON *j, Params *p) {
    for (i64 i = 0; i < N_PARAMS; i++) {
        json.matrix(json.get(j, PARAM_NAMES[i]), p->all[i]->data, p->all[i]->shape[0], p->all[i]->shape[1], i >= 2);
    }
}

// json numbers are doubles, token ids are ints
static const i64 *load_ids(const cJSON *j, Arena *data) {
    f64 *raw = arena.alloc(data, NUM_STEPS * BLOCK_SIZE * sizeof(f64));
    i64 *ids = arena.alloc(data, NUM_STEPS * BLOCK_SIZE * sizeof(i64));
    json.matrix(j, raw, NUM_STEPS, BLOCK_SIZE, false);
    for (i64 i = 0; i < NUM_STEPS * BLOCK_SIZE; i++) {
        ids[i] = (i64)raw[i];
    }
    return ids;
}

// every final weight within 1e-11 of the reference
static void check_parity(const Params *p, const Params *ref) {
    f64 worst = 0.0;
    for (i64 i = 0; i < N_PARAMS; i++) {
        worst = fmax(worst, check.allclose(p->all[i]->data, ref->all[i]->data, p->all[i]->size, 0.0, 1e-11, PARAM_NAMES[i]));
    }
    printf("passed parity check (max abs diff %.2g)\n", worst);
}

i32 main(void) {
    setvbuf(stdout, NULL, _IONBF, 0); // unbuffered, so stdio never mallocs a buffer

    // all memory, sized at compile time
    static alignas(16) u8 perm_buf[512 * 1024];      // two param sets, their grads and AdamW state use about 200 KB
    static alignas(16) u8 graph_buf[256 * 1024];     // one loss plus backward uses about 225 KB
    static alignas(16) u8 data_buf[6 * 1024 * 1024]; // parity.json, its parse tree and the decoded batches use about 4.8 MB
    Mem mem = {.perm = {.buf = perm_buf, .cap = sizeof(perm_buf)}, .graph = {.buf = graph_buf, .cap = sizeof(graph_buf)}};
    Arena data = {.buf = data_buf, .cap = sizeof(data_buf)};

    // decoded once before training, so the training loop does no json work
    const cJSON *j = json.load("parity.json", &data);
    assert((i64)json.get(j, "bos")->valuedouble == BOS);
    const i64 *inputs = load_ids(json.get(j, "inputs"), &data);
    const i64 *targets = load_ids(json.get(j, "targets"), &data);
    f64 *masks = arena.alloc(&data, NUM_STEPS * BLOCK_SIZE * sizeof(f64));
    json.matrix(json.get(j, "masks"), masks, NUM_STEPS, BLOCK_SIZE, false);
    Params p = microgpt.params(&mem), ref = microgpt.params(&mem);
    load_params(json.get(j, "init_weights"), &p);
    load_params(json.get(j, "final_weights"), &ref);

    AdamW opt = adamw.new(p.all, N_PARAMS);
    clock_t start = clock();
    for (i64 step = 0; step < NUM_STEPS; step++) {
        i64 at = step * BLOCK_SIZE;
        tensor.backward(microgpt.loss(&p, &inputs[at], &targets[at], &masks[at], BLOCK_SIZE));
        tensor.reset_graph(&mem);
        adamw.step(&opt, step + 1, NUM_STEPS);
    }
    printf("trained %ld steps in %.3fs\n", (long)NUM_STEPS, (f64)(clock() - start) / CLOCKS_PER_SEC);

    check_parity(&p, &ref);

    char name[BLOCK_SIZE] = {0};
    microgpt.sample(&p, (u64)time(NULL), name);
    printf("inference (random sampled name): %s\n", name);

    return 0;
}
