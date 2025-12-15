#include "optimizers.h"
#include "tensor.h"
#include "unity.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

void test_optimizer_zero_grad(void) {
    uint64_t shape[] = {2, 2};
    Tensor *w = tensor_create((float32_t[]){1.0f, 2.0f, 3.0f, 4.0f}, shape, 2, true);
    Tensor *grad = tensor_create((float32_t[]){0.1f, 0.2f, 0.3f, 0.4f}, shape, 2, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.0f, 0.0f);

    TEST_ASSERT_NOT_NULL(w->grad);
    optimizer_zero_grad(opt);
    TEST_ASSERT_NULL(w->grad);

    optimizer_free(opt);
    tensor_free(w);
}

void test_sgd_step_basic(void) {
    uint64_t shape[] = {2};
    Tensor *w = tensor_create((float32_t[]){1.0f, 2.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){0.1f, 0.2f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.0f, 0.0f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.99f, w->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.98f, w->data[1]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_sgd_step_momentum(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);

    Tensor *grad1 = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad1;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.9f, 0.0f);

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.1f, w->data[0]);

    tensor_free(grad1);
    w->grad = NULL;

    Tensor *grad2 = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad2;

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.29f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_sgd_step_weight_decay(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){2.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.0f, 0.01f);

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.898f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_step_basic(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adam_create(params, 1, 0.1f, 0.9f, 0.999f, 0.0f, 0.0f);

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.1f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_step_multiple(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adam_create(params, 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f);

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.1f, w->data[0]);

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.2f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_with_weight_decay(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){2.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adam_create(params, 1, 0.1f, 0.9f, 0.999f, 0.0f, 0.1f);

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.9f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adamw_step_basic(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adamw_create(params, 1, 0.1f, 0.9f, 0.999f, 0.0f, 0.0f);

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.1f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adamw_decoupled_decay(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){2.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){0.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};

    Optimizer *opt = optimizer_adamw_create(params, 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.1f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.98f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_sgd_multiple_params(void) {
    uint64_t shape[] = {1};
    Tensor *w1 = tensor_create((float32_t[]){1.0f}, shape, 1, true);
    Tensor *w2 = tensor_create((float32_t[]){2.0f}, shape, 1, true);

    Tensor *g1 = tensor_create((float32_t[]){0.1f}, shape, 1, false);
    Tensor *g2 = tensor_create((float32_t[]){0.2f}, shape, 1, false);

    w1->grad = g1;
    w2->grad = g2;

    Tensor *params[] = {w1, w2};
    Optimizer *opt = optimizer_sgd_create(params, 2, 0.1f, 0.0f, 0.0f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.99f, w1->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.98f, w2->data[0]);

    optimizer_free(opt);
    tensor_free(w1);
    tensor_free(w2);
}

void test_sgd_zero_learning_rate(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){10.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.0f, 0.9f, 0.1f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 10.0f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_sgd_high_momentum(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.99f, 0.0f);

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.1f, w->data[0]);

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.299f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_alternate_betas(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adam_create(params, 1, 0.1f, 0.5f, 0.5f, 1e-8f, 0.0f);

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.1f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_numerical_stability(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1e-20f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adam_create(params, 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f);

    optimizer_step(opt);

    TEST_ASSERT_TRUE(!isnan(w->data[0]));
    TEST_ASSERT_TRUE(!isinf(w->data[0]));

    optimizer_free(opt);
    tensor_free(w);
}

void test_optimizer_large_tensor(void) {
    size_t size = 1000;
    uint64_t shape[] = {size};
    float32_t *data = calloc(size, sizeof(float32_t));
    float32_t *g_data = calloc(size, sizeof(float32_t));
    for (size_t i = 0; i < size; ++i) {
        data[i] = 1.0f;
        g_data[i] = 0.1f;
    }

    Tensor *w = tensor_create(data, shape, 1, true);
    Tensor *grad = tensor_create(g_data, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.0f, 0.0f);

    optimizer_step(opt);

    for (size_t i = 0; i < size; i += 100) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.99f, w->data[i]);
    }

    free(data);
    free(g_data);
    optimizer_free(opt);
    tensor_free(w);
}

void test_optimizer_skip_null_grad(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){1.0f}, shape, 1, true);

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.0f, 0.0f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_sgd_integration(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){2.0f}, shape, 1, true);
    Tensor *g = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = g;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.9f, 0.1f);

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.88f, w->data[0]);

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.6532f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_vs_adamw_difference(void) {
    uint64_t shape[] = {1};

    Tensor *w_adam = tensor_create((float32_t[]){2.0f}, shape, 1, true);
    Tensor *g_adam = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w_adam->grad = g_adam;

    Tensor *w_adamw = tensor_create((float32_t[]){2.0f}, shape, 1, true);
    Tensor *g_adamw = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w_adamw->grad = g_adamw;

    Tensor *p_adam[] = {w_adam};
    Tensor *p_adamw[] = {w_adamw};

    float32_t wd = 0.1f;
    float32_t lr = 0.1f;

    Optimizer *opt_adam = optimizer_adam_create(p_adam, 1, lr, 0.9f, 0.999f, 1e-8f, wd);
    Optimizer *opt_adamw = optimizer_adamw_create(p_adamw, 1, lr, 0.9f, 0.999f, 1e-8f, wd);

    optimizer_step(opt_adam);
    optimizer_step(opt_adamw);

    TEST_ASSERT_FALSE(w_adam->data[0] == w_adamw->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, 1.900f, w_adam->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, 1.881f, w_adamw->data[0]);

    optimizer_free(opt_adam);
    optimizer_free(opt_adamw);
    tensor_free(w_adam);
    tensor_free(w_adamw);
}

void test_sgd_large_lr(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){1.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 10.0f, 0.0f, 0.0f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -9.0f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_sgd_negative_lr(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){1.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, -0.1f, 0.0f, 0.0f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.1f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_sgd_momentum_accumulation(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.5f, 0.0f);

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.1f, w->data[0]);

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.25f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_sgd_momentum_zero_grad_updates(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.9f, 0.0f);

    optimizer_step(opt);

    memset(grad->data, 0, grad->size * sizeof(float32_t));

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.19f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_sgd_high_weight_decay(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){10.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){0.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.0f, 0.5f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 9.5f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_sgd_tiny_grad(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1e-20f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 1.0f, 0.0f, 0.0f);

    optimizer_step(opt);

    TEST_ASSERT_NOT_EQUAL(0.0f, w->data[0]);
    TEST_ASSERT_TRUE(!isnan(w->data[0]));

    optimizer_free(opt);
    tensor_free(w);
}

void test_sgd_tiny_param(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){1e-20f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){0.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.0f, 1.0f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-40f, 0.9e-20f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_beta1_zero(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adam_create(params, 1, 0.1f, 0.0f, 0.999f, 1e-8f, 0.0f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.1f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_beta1_high(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adam_create(params, 1, 0.1f, 0.99f, 0.999f, 1e-8f, 0.0f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.1f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_beta2_min(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adam_create(params, 1, 0.1f, 0.9f, 0.0f, 1e-8f, 0.0f);

    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.1f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_beta2_high(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adam_create(params, 1, 0.1f, 0.9f, 0.9999f, 1e-8f, 0.0f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.1f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_large_epsilon(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adam_create(params, 1, 0.1f, 0.9f, 0.999f, 1.0f, 0.0f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.05f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_step_count_increment(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adam_create(params, 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f);

    TEST_ASSERT_EQUAL_UINT64(0, opt->step_count);
    optimizer_step(opt);
    TEST_ASSERT_EQUAL_UINT64(1, opt->step_count);
    optimizer_step(opt);
    TEST_ASSERT_EQUAL_UINT64(2, opt->step_count);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_nan_grad(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){NAN}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adam_create(params, 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f);

    optimizer_step(opt);

    TEST_ASSERT_TRUE(isnan(w->data[0]));

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_inf_grad(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){INFINITY}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adam_create(params, 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f);

    optimizer_step(opt);

    TEST_ASSERT_TRUE(isnan(w->data[0]) || isinf(w->data[0]));

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_param_recovery(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){100.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){0.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};

    Optimizer *opt = optimizer_adam_create(params, 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.1f);

    optimizer_step(opt);

    TEST_ASSERT_TRUE(w->data[0] < 100.0f);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adamw_no_decay_vs_adam_no_decay(void) {
    uint64_t shape[] = {1};
    Tensor *w1 = tensor_create((float32_t[]){1.0f}, shape, 1, true);
    Tensor *w2 = tensor_create((float32_t[]){1.0f}, shape, 1, true);
    Tensor *g1 = tensor_create((float32_t[]){0.1f}, shape, 1, false);
    Tensor *g2 = tensor_create((float32_t[]){0.1f}, shape, 1, false);
    w1->grad = g1;
    w2->grad = g2;

    Tensor *p1[] = {w1};
    Tensor *p2[] = {w2};

    Optimizer *opt1 = optimizer_adam_create(p1, 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f);
    Optimizer *opt2 = optimizer_adamw_create(p2, 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f);

    optimizer_step(opt1);
    optimizer_step(opt2);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, w1->data[0], w2->data[0]);

    optimizer_free(opt1);
    optimizer_free(opt2);
    tensor_free(w1);
    tensor_free(w2);
}

void test_adamw_high_decay(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){10.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){0.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adamw_create(params, 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.5f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 9.5f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adamw_negative_decay(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){1.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){0.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adamw_create(params, 1, 0.1f, 0.9f, 0.999f, 1e-8f, -0.1f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.01f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adamw_step_accumulation(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adamw_create(params, 1, 0.1f, 0.9f, 0.999f, 1e-8f, 0.0f);

    optimizer_step(opt);
    float32_t v1 = w->data[0];
    optimizer_step(opt);
    float32_t v2 = w->data[0];

    TEST_ASSERT_TRUE(fabs(v2 - v1) > fabs(v1));

    optimizer_free(opt);
    tensor_free(w);
}

void test_optimizer_zero_grad_all_null(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){1.0f}, shape, 1, true);

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.0f, 0.0f);

    optimizer_zero_grad(opt);

    optimizer_free(opt);
    tensor_free(w);
}

void test_optimizer_zero_grad_mixed(void) {
    uint64_t shape[] = {1};
    Tensor *w1 = tensor_create((float32_t[]){1.0f}, shape, 1, true);
    Tensor *w2 = tensor_create((float32_t[]){1.0f}, shape, 1, true);

    Tensor *g2 = tensor_create((float32_t[]){0.1f}, shape, 1, false);
    w2->grad = g2;

    Tensor *params[] = {w1, w2};
    Optimizer *opt = optimizer_sgd_create(params, 2, 0.1f, 0.0f, 0.0f);

    TEST_ASSERT_NULL(w1->grad);
    TEST_ASSERT_NOT_NULL(w2->grad);

    optimizer_zero_grad(opt);

    TEST_ASSERT_NULL(w1->grad);
    TEST_ASSERT_NULL(w2->grad);

    optimizer_free(opt);
    tensor_free(w1);
    tensor_free(w2);
}

void test_optimizer_creation_defaults_sgd(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){1.0f}, shape, 1, true);
    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.0f, 0.0f);
    TEST_ASSERT_NOT_NULL(opt);
    optimizer_free(opt);
    tensor_free(w);
}

void test_optimizer_creation_defaults_adam(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){1.0f}, shape, 1, true);
    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adam_create(params, 1, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);
    TEST_ASSERT_NOT_NULL(opt);
    optimizer_free(opt);
    tensor_free(w);
}

void test_mixed_shapes(void) {
    uint64_t shape1[] = {1};
    uint64_t shape2[] = {2};
    Tensor *w1 = tensor_create((float32_t[]){1.0f}, shape1, 1, true);
    Tensor *w2 = tensor_create((float32_t[]){1.0f, 2.0f}, shape2, 1, true);

    Tensor *g1 = tensor_create((float32_t[]){1.0f}, shape1, 1, false);
    Tensor *g2 = tensor_create((float32_t[]){0.1f, 0.2f}, shape2, 1, false);

    w1->grad = g1;
    w2->grad = g2;

    Tensor *params[] = {w1, w2};
    Optimizer *opt = optimizer_sgd_create(params, 2, 0.1f, 0.0f, 0.0f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.9f, w1->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.99f, w2->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.98f, w2->data[1]);

    optimizer_free(opt);
    tensor_free(w1);
    tensor_free(w2);
}

void test_large_batch_simulation(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){10.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){0.1f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.01f, 0.0f, 0.0f);

    for (int i = 0; i < 100; ++i) {
        optimizer_step(opt);
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 9.9f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_shapes_independent(void) {
    uint64_t s1[] = {2, 2};
    uint64_t s2[] = {4};

    Tensor *w1 = tensor_create((float32_t[]){1, 2, 3, 4}, s1, 2, true);
    Tensor *w2 = tensor_create((float32_t[]){1, 2, 3, 4}, s2, 1, true);

    Tensor *g1 = tensor_create((float32_t[]){1, 1, 1, 1}, s1, 2, false);
    Tensor *g2 = tensor_create((float32_t[]){1, 1, 1, 1}, s2, 1, false);

    w1->grad = g1;
    w2->grad = g2;

    Tensor *params[] = {w1, w2};
    Optimizer *opt = optimizer_sgd_create(params, 2, 0.1f, 0.0f, 0.0f);

    optimizer_step(opt);

    for (int i = 0; i < 4; ++i) {
        TEST_ASSERT_EQUAL_FLOAT(w1->data[i], w2->data[i]);
    }

    optimizer_free(opt);
    tensor_free(w1);
    tensor_free(w2);
}

void test_zero_grad_check_values(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){1.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.0f, 0.0f);

    optimizer_zero_grad(opt);

    TEST_ASSERT_NULL(w->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_bias_correction_step1(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adam_create(params, 1, 0.1f, 0.9f, 0.999f, 0.0f, 0.0f);

    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, -0.1f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_bias_correction_step2(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = grad;

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_adam_create(params, 1, 0.1f, 0.9f, 0.999f, 0.0f, 0.0f);

    optimizer_step(opt);
    optimizer_step(opt);

    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -0.2f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_optimizer_zero_grad);
    RUN_TEST(test_sgd_step_basic);
    RUN_TEST(test_sgd_step_momentum);
    RUN_TEST(test_sgd_step_weight_decay);
    RUN_TEST(test_adam_step_basic);
    RUN_TEST(test_adam_step_multiple);
    RUN_TEST(test_adam_with_weight_decay);
    RUN_TEST(test_adamw_step_basic);
    RUN_TEST(test_adamw_decoupled_decay);
    RUN_TEST(test_sgd_multiple_params);
    RUN_TEST(test_sgd_zero_learning_rate);
    RUN_TEST(test_sgd_high_momentum);
    RUN_TEST(test_adam_alternate_betas);
    RUN_TEST(test_adam_numerical_stability);
    RUN_TEST(test_optimizer_large_tensor);
    RUN_TEST(test_optimizer_skip_null_grad);
    RUN_TEST(test_sgd_integration);
    RUN_TEST(test_adam_vs_adamw_difference);
    RUN_TEST(test_sgd_large_lr);
    RUN_TEST(test_sgd_negative_lr);
    RUN_TEST(test_sgd_momentum_accumulation);
    RUN_TEST(test_sgd_momentum_zero_grad_updates);
    RUN_TEST(test_sgd_high_weight_decay);
    RUN_TEST(test_sgd_tiny_grad);
    RUN_TEST(test_sgd_tiny_param);
    RUN_TEST(test_adam_beta1_zero);
    RUN_TEST(test_adam_beta1_high);
    RUN_TEST(test_adam_beta2_min);
    RUN_TEST(test_adam_beta2_high);
    RUN_TEST(test_adam_large_epsilon);
    RUN_TEST(test_adam_step_count_increment);
    RUN_TEST(test_adam_nan_grad);
    RUN_TEST(test_adam_inf_grad);
    RUN_TEST(test_adam_param_recovery);
    RUN_TEST(test_adamw_no_decay_vs_adam_no_decay);
    RUN_TEST(test_adamw_high_decay);
    RUN_TEST(test_adamw_negative_decay);
    RUN_TEST(test_adamw_step_accumulation);
    RUN_TEST(test_optimizer_zero_grad_all_null);
    RUN_TEST(test_optimizer_zero_grad_mixed);
    RUN_TEST(test_optimizer_creation_defaults_sgd);
    RUN_TEST(test_optimizer_creation_defaults_adam);
    RUN_TEST(test_mixed_shapes);
    RUN_TEST(test_large_batch_simulation);
    RUN_TEST(test_shapes_independent);
    RUN_TEST(test_zero_grad_check_values);
    RUN_TEST(test_adam_bias_correction_step1);
    RUN_TEST(test_adam_bias_correction_step2);
    return UNITY_END();
}
