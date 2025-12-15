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

    // Cleanup old grad before replacing it, although w->grad owns it.
    // If we want to replace w->grad, we should free the old one first OR ensure w->grad is what we think.
    // Here, w->grad IS grad1.
    // If we want to set a new gradient for step 2:

    // 1. Manually free old grad because we are about to overwrite the pointer.
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
    return UNITY_END();
}
