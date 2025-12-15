#include "optimizers.h"
#include "tensor.h"
#include "unity.h"
#include <math.h>
#include <stdlib.h>

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
    return UNITY_END();
}
