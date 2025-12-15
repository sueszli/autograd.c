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

void test_optimizer_skip_null_grad(void) {
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){1.0f}, shape, 1, true);
    // No gradient assigned: w->grad is NULL by default

    Tensor *params[] = {w};
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.0f, 0.0f);

    optimizer_step(opt);

    // Should be unchanged
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.0f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_sgd_integration(void) {
    // Test interplay of momentum and weight decay
    uint64_t shape[] = {1};
    Tensor *w = tensor_create((float32_t[]){2.0f}, shape, 1, true);
    Tensor *g = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w->grad = g;

    Tensor *params[] = {w};
    // lr=0.1, momentum=0.9, weight_decay=0.1
    Optimizer *opt = optimizer_sgd_create(params, 1, 0.1f, 0.9f, 0.1f);

    // Step 1
    // g_wd = 1.0 + 0.1*2.0 = 1.2
    // v = 0.9*0 + 1.2 = 1.2
    // w = 2.0 - 0.1*1.2 = 1.88
    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.88f, w->data[0]);

    // Step 2
    // w is 1.88. Grad is still 1.0 (manually managed for test)
    // g_wd = 1.0 + 0.1*1.88 = 1.188
    // v = 0.9*1.2 + 1.188 = 1.08 + 1.188 = 2.268
    // w = 1.88 - 0.1*2.268 = 1.88 - 0.2268 = 1.6532
    optimizer_step(opt);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 1.6532f, w->data[0]);

    optimizer_free(opt);
    tensor_free(w);
}

void test_adam_vs_adamw_difference(void) {
    uint64_t shape[] = {1};
    
    // Setup for Adam
    Tensor *w_adam = tensor_create((float32_t[]){2.0f}, shape, 1, true);
    Tensor *g_adam = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w_adam->grad = g_adam;
    
    // Setup for AdamW
    Tensor *w_adamw = tensor_create((float32_t[]){2.0f}, shape, 1, true);
    Tensor *g_adamw = tensor_create((float32_t[]){1.0f}, shape, 1, false);
    w_adamw->grad = g_adamw;

    Tensor *p_adam[] = {w_adam};
    Tensor *p_adamw[] = {w_adamw};

    // Use same high weight decay to make difference obvious
    float32_t wd = 0.1f;
    float32_t lr = 0.1f;
    
    Optimizer *opt_adam = optimizer_adam_create(p_adam, 1, lr, 0.9f, 0.999f, 1e-8f, wd);
    Optimizer *opt_adamw = optimizer_adamw_create(p_adamw, 1, lr, 0.9f, 0.999f, 1e-8f, wd);

    optimizer_step(opt_adam);
    optimizer_step(opt_adamw);

    // Adam: WD added to grad. Gradient becomes 1.2.
    // m ~ 0.12, v ~ 0.00144. m_hat ~ 1.2, v_hat ~ 1.44.
    // update ~ 0.1 * 1.2 / 1.2 = 0.1.
    // w ~ 1.9.
    
    // AdamW:
    // m ~ 0.1, v ~ 0.001. m_hat ~ 1.0, v_hat ~ 1.0. 
    // update ~ 0.1 * 1.0 / 1.0 = 0.1.
    // w = (2.0 - 0.1) * (1 - 0.01) = 1.9 * 0.99 = 1.881
    
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
    RUN_TEST(test_optimizer_skip_null_grad);
    RUN_TEST(test_sgd_integration);
    RUN_TEST(test_adam_vs_adamw_difference);
    return UNITY_END();
}
