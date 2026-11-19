#include "autograd.h"
#include "cJSON.h"
#include "microgpt.h"
#include "optim.h"
#include "std/mem.h"
#include "std/testing.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define NUM_STEPS 1000

static const char *NAMES[N_PARAMS] = {"wte", "wpe", "lm_head", "layer0.attn_wq", "layer0.attn_wk", "layer0.attn_wv", "layer0.attn_wo", "layer0.mlp_fc1", "layer0.mlp_fc2"};

static cJSON *field(const cJSON *j, const char *key) {
    cJSON *f = cJSON_GetObjectItemCaseSensitive(j, key);
    assert(f != NULL);
    return f;
}

// the reference stores linear weights as (out, in), wte and wpe are already (rows, cols)
static void load_params(const cJSON *j, Params *p) {
    for (int64_t i = 0; i < N_PARAMS; i++) {
        Tensor *t = p->all[i];
        const cJSON *rows = field(j, NAMES[i]);
        bool transposed = i >= 2;
        assert(cJSON_GetArraySize(rows) == (transposed ? t->cols : t->rows));
        for (int64_t r = 0; r < cJSON_GetArraySize(rows); r++) {
            const cJSON *row = cJSON_GetArrayItem(rows, (int)r);
            assert(cJSON_GetArraySize(row) == (transposed ? t->rows : t->cols));
            for (int64_t c = 0; c < cJSON_GetArraySize(row); c++) {
                t->data[transposed ? c * t->cols + r : r * t->cols + c] = cJSON_GetArrayItem(row, (int)c)->valuedouble;
            }
        }
    }
}

// decoded once before training, so the training loop does no json work
static int64_t inputs[NUM_STEPS][BLOCK_SIZE], targets[NUM_STEPS][BLOCK_SIZE];
static double masks[NUM_STEPS][BLOCK_SIZE];

static void load_rows(const cJSON *rows, int64_t (*ints)[BLOCK_SIZE], double (*doubles)[BLOCK_SIZE]) {
    assert(cJSON_GetArraySize(rows) >= NUM_STEPS && "not enough sequences");
    const cJSON *row = rows->child;
    for (int64_t step = 0; step < NUM_STEPS; step++, row = row->next) {
        assert(cJSON_GetArraySize(row) == BLOCK_SIZE && "sequence length must be BLOCK_SIZE");
        const cJSON *v = row->child;
        for (int64_t i = 0; i < BLOCK_SIZE; i++, v = v->next) {
            if (ints != NULL) {
                ints[step][i] = (int64_t)v->valuedouble;
            } else {
                doubles[step][i] = v->valuedouble;
            }
        }
    }
}

// cJSON and the raw file get their own static buffers
// cJSON frees without a length, so free does nothing
static alignas(16) unsigned char json_buf[4 * 1024 * 1024]; // parity.json parses into about 3.8 MB
static Arena json_mem = {.buf = json_buf, .cap = sizeof(json_buf)};
static void *json_alloc(size_t bytes) { return arena_alloc(&json_mem, bytes); }
static void json_free(void *p) { (void)p; }

static cJSON *read_json(const char *path) {
    static char raw[512 * 1024];
    FILE *f = fopen(path, "rb");
    assert(f != NULL && "cannot open parity.json, run from the repo root");
    size_t size = fread(raw, 1, sizeof(raw) - 1, f);
    assert(feof(f) && size > 0 && "cannot read parity.json, or it outgrew its buffer");
    int close_ok = fclose(f);
    assert(close_ok == 0 && "cannot close parity.json");
    cJSON_InitHooks(&(cJSON_Hooks){.malloc_fn = json_alloc, .free_fn = json_free});
    cJSON *j = cJSON_Parse(raw);
    assert(j != NULL && "cannot parse parity.json");
    return j;
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
    cJSON *j = read_json("parity.json");
    load_rows(field(j, "inputs"), inputs, NULL);
    load_rows(field(j, "targets"), targets, NULL);
    load_rows(field(j, "masks"), NULL, masks);
    Params p = params_new(), ref = params_new();
    load_params(field(j, "init_weights"), &p);
    load_params(field(j, "final_weights"), &ref);

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
    assert(field(j, "bos")->valuedouble == BOS);
    sample(&p, (uint64_t)time(NULL), name);
    printf("inference (random sampled name): %s\n", name);

    return 0;
}
