#include "autograd.h"
#include "cJSON.h"
#include "microgpt.h"
#include "optim.h"
#include <assert.h>
#include <math.h>
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

static cJSON *read_json(const char *path) {
    FILE *f = fopen(path, "rb");
    assert(f != NULL && "cannot open parity.json, run from the repo root");
    int seek_ok = fseek(f, 0, SEEK_END);
    long size = ftell(f);
    assert(seek_ok == 0 && size > 0 && "cannot size parity.json");
    rewind(f);
    char *raw = malloc((size_t)size + 1);
    assert(raw != NULL && "malloc failed");
    size_t read = fread(raw, 1, (size_t)size, f);
    assert(read == (size_t)size && "cannot read parity.json");
    raw[size] = '\0';
    int close_ok = fclose(f);
    assert(close_ok == 0 && "cannot close parity.json");
    cJSON *j = cJSON_Parse(raw);
    free(raw);
    assert(j != NULL && "cannot parse parity.json");
    return j;
}

// returns the number of tensors further than atol from the reference
static int64_t check_parity(const Params *p, const Params *ref) {
    const double atol = 1e-11;
    int64_t failures = 0;
    double worst = 0.0;
    for (int64_t i = 0; i < N_PARAMS; i++) {
        double max_diff = 0.0;
        for (int64_t k = 0; k < p->all[i]->rows * p->all[i]->cols; k++) {
            max_diff = fmax(max_diff, fabs(p->all[i]->data[k] - ref->all[i]->data[k]));
        }
        worst = fmax(worst, max_diff);
        if (max_diff > atol) {
            fprintf(stderr, "  %s: max abs diff = %g\n", NAMES[i], max_diff);
            failures++;
        }
    }
    if (failures == 0) {
        printf("passed parity check (max abs diff %.2g)\n", worst);
    } else {
        fprintf(stderr, "FAIL: %ld/%d tensors exceed atol %g\n", (long)failures, N_PARAMS, atol);
    }
    return failures;
}

// the boundary token both starts and ends a name
static void sample_name(const Params *p, int64_t bos, char name[BLOCK_SIZE]) {
    uint64_t seed = (uint64_t)time(NULL);
    int64_t tokens[BLOCK_SIZE] = {bos};
    for (int64_t n = 1; n < BLOCK_SIZE; n++) {
        const Tensor *out = logits(p, tokens, n);
        double probs[VOCAB_SIZE], max = out->data[(n - 1) * VOCAB_SIZE], sum = 0.0;
        for (int64_t c = 0; c < VOCAB_SIZE; c++) {
            max = fmax(max, out->data[(n - 1) * VOCAB_SIZE + c]);
        }
        for (int64_t c = 0; c < VOCAB_SIZE; c++) {
            probs[c] = exp(out->data[(n - 1) * VOCAB_SIZE + c] - max);
            sum += probs[c];
        }
        autograd_reset();
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; // 64-bit lcg, as in autograd.lean
        double u = (double)(seed >> 11) / 9007199254740992.0 * sum;    // uniform in [0, sum)
        int64_t next = VOCAB_SIZE - 1;
        for (int64_t c = 0; c < VOCAB_SIZE; c++) {
            u -= probs[c];
            if (u < 0.0) {
                next = c;
                break;
            }
        }
        if (next == bos) {
            return;
        }
        tokens[n] = next;
        name[n - 1] = (char)('a' + next);
    }
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
        autograd_reset();
        adamw_step(&opt, step + 1, NUM_STEPS);
    }
    printf("trained %d steps in %.3fs\n", NUM_STEPS, (double)(clock() - start) / CLOCKS_PER_SEC);

    int64_t failures = check_parity(&p, &ref);

    char name[BLOCK_SIZE] = {0};
    sample_name(&p, (int64_t)field(j, "bos")->valuedouble, name);
    printf("inference (random sampled name): %s\n", name);

    adamw_free(&opt);
    params_free(&p);
    params_free(&ref);
    cJSON_Delete(j);
    return failures == 0 ? 0 : 1;
}
