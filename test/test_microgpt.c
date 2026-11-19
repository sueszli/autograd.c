#include "microgpt.h"
#include "optim.h"
#include "std/assert.h"
#include "std/random.h"
#include "tensor.h"
#include "unity.h"
#include <math.h>

static alignas(16) u8 perm_buf[1 << 20], graph_buf[1 << 20];
static Mem mem;

void setUp(void) { mem = (Mem){.perm = {.buf = perm_buf, .cap = sizeof(perm_buf)}, .graph = {.buf = graph_buf, .cap = sizeof(graph_buf)}}; }
void tearDown(void) {}

static u64 seed = 1;

static const i64 ids[] = {26, 3, 1, 4, 26, 0};
static const i64 targets[] = {3, 1, 4, 26, 0, 0};
static const f64 mask[] = {1, 1, 1, 1, 0, 0};

void test_zero_weights_give_uniform_loss(void) {
    Params p = microgpt.params(&mem);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, log(VOCAB_SIZE), microgpt.loss(&p, ids, targets, mask, 6)->data[0]);
}

void test_logits_shape(void) {
    Params p = microgpt.params(&mem);
    Tensor *out = microgpt.logits(&p, ids, 3);
    TEST_ASSERT_EQUAL_INT64(3, out->shape[0]);
    TEST_ASSERT_EQUAL_INT64(VOCAB_SIZE, out->shape[1]);
}

static f64 loss_of(void *p) {
    f64 l = microgpt.loss(p, ids, targets, mask, 6)->data[0];
    tensor.reset_graph(&mem);
    return l;
}

// every weight of the full model, checked with assert_grad
void test_loss_gradcheck(void) {
    Params p = microgpt.params(&mem);
    for (i64 i = 0; i < N_PARAMS; i++) {
        rng.uniform(&seed, p.all[i]->data, p.all[i]->size, -0.3, 0.3);
    }
    tensor.backward(microgpt.loss(&p, ids, targets, mask, 6));
    tensor.reset_graph(&mem);
    for (i64 i = 0; i < N_PARAMS; i++) {
        check.grad(loss_of, &p, p.all[i]->data, p.all[i]->grad, p.all[i]->size, 1e-6, 1e-6, "grad");
    }
}

// step 1: m = 0.15 g, v = 0.01 g^2, both bias corrections cancel, so p -= lr * g / |g|
void test_adamw_first_step(void) {
    Tensor *w = tensor.new(&mem, shape(1, 2), (f64[]){1, -3}, true);
    AdamW opt = adamw.new(&w, 1);
    w->grad[0] = 2.0;
    adamw.step(&opt, 1, 1000);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, 0.3, opt.m[0][0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, 0.04, opt.v[0][0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-9, 1 - 0.01, w->data[0]);
    TEST_ASSERT_EQUAL_DOUBLE(-3, w->data[1]);
    TEST_ASSERT_EQUAL_DOUBLE(0, w->grad[0]);
}

// a few hundred steps on one sequence should drive the loss well below uniform
void test_training_reduces_loss(void) {
    Params p = microgpt.params(&mem);
    for (i64 i = 0; i < N_PARAMS; i++) {
        rng.uniform(&seed, p.all[i]->data, p.all[i]->size, -0.08, 0.08);
    }
    AdamW opt = adamw.new(p.all, N_PARAMS);
    f64 last = 0.0;
    for (i64 step = 1; step <= 200; step++) {
        Tensor *l = microgpt.loss(&p, ids, targets, mask, 6);
        last = l->data[0];
        tensor.backward(l);
        tensor.reset_graph(&mem);
        adamw.step(&opt, step, 200);
    }
    TEST_ASSERT_LESS_THAN_DOUBLE(0.1, last);
}

i32 main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_zero_weights_give_uniform_loss);
    RUN_TEST(test_logits_shape);
    RUN_TEST(test_loss_gradcheck);
    RUN_TEST(test_adamw_first_step);
    RUN_TEST(test_training_reduces_loss);
    return UNITY_END();
}
