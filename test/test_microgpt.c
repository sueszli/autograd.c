#include "microgpt.h"
#include "optim.h"
#include "std/assert.h"
#include "std/random.h"
#include "tensor.h"
#include "unity.h"
#include <math.h>

void setUp(void) {}
void tearDown(void) {
    graph_reset();
    perm_reset();
}

static uint64_t rng = 1;

static const int64_t ids[] = {26, 3, 1, 4, 26, 0};
static const int64_t targets[] = {3, 1, 4, 26, 0, 0};
static const double mask[] = {1, 1, 1, 1, 0, 0};

void test_zero_weights_give_uniform_loss(void) {
    Params p = params_new();
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, log(VOCAB_SIZE), loss(&p, ids, targets, mask, 6)->data[0]);
}

void test_logits_shape(void) {
    Params p = params_new();
    Tensor *out = logits(&p, ids, 3);
    TEST_ASSERT_EQUAL_INT64(3, out->shape[0]);
    TEST_ASSERT_EQUAL_INT64(VOCAB_SIZE, out->shape[1]);
}

static double loss_of(void *p) {
    double l = loss(p, ids, targets, mask, 6)->data[0];
    graph_reset();
    return l;
}

// every weight of the full model, checked with assert_grad
void test_loss_gradcheck(void) {
    Params p = params_new();
    for (int64_t i = 0; i < N_PARAMS; i++) {
        random_uniform(&rng, p.all[i]->data, p.all[i]->size, -0.3, 0.3);
    }
    backward(loss(&p, ids, targets, mask, 6));
    graph_reset();
    for (int64_t i = 0; i < N_PARAMS; i++) {
        assert_grad(loss_of, &p, p.all[i]->data, p.all[i]->grad, p.all[i]->size, 1e-6, 1e-6, "grad");
    }
}

// step 1: m = 0.15 g, v = 0.01 g^2, both bias corrections cancel, so p -= lr * g / |g|
void test_adamw_first_step(void) {
    Tensor *w = tensor_new(SHAPE(1, 2), (double[]){1, -3}, true);
    AdamW opt = adamw_new(&w, 1);
    w->grad[0] = 2.0;
    adamw_step(&opt, 1, 1000);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, 0.3, opt.m[0][0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, 0.04, opt.v[0][0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-9, 1 - 0.01, w->data[0]);
    TEST_ASSERT_EQUAL_DOUBLE(-3, w->data[1]);
    TEST_ASSERT_EQUAL_DOUBLE(0, w->grad[0]);
}

// a few hundred steps on one sequence should drive the loss well below uniform
void test_training_reduces_loss(void) {
    Params p = params_new();
    for (int64_t i = 0; i < N_PARAMS; i++) {
        random_uniform(&rng, p.all[i]->data, p.all[i]->size, -0.08, 0.08);
    }
    AdamW opt = adamw_new(p.all, N_PARAMS);
    double last = 0.0;
    for (int64_t step = 1; step <= 200; step++) {
        Tensor *l = loss(&p, ids, targets, mask, 6);
        last = l->data[0];
        backward(l);
        graph_reset();
        adamw_step(&opt, step, 200);
    }
    TEST_ASSERT_LESS_THAN_DOUBLE(0.1, last);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_zero_weights_give_uniform_loss);
    RUN_TEST(test_logits_shape);
    RUN_TEST(test_loss_gradcheck);
    RUN_TEST(test_adamw_first_step);
    RUN_TEST(test_training_reduces_loss);
    return UNITY_END();
}
