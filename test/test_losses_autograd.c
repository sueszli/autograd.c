#include "autograd.h"
#include "ops/arithmetic.h"
#include "ops/losses.h"
#include "tensor.h"
#include "unity.h"
#include <float.h>
#include <math.h>

void setUp(void) {}
void tearDown(void) {}

static Tensor *create_tensor_1d(float32_t *data, uint64_t size, bool requires_grad) {
    uint64_t shape[] = {size};
    return tensor_create(data, shape, 1, requires_grad);
}

static Tensor *create_tensor_2d(float32_t *data, uint64_t rows, uint64_t cols, bool requires_grad) {
    uint64_t shape[] = {rows, cols};
    return tensor_create(data, shape, 2, requires_grad);
}

void test_mse_loss_backward_simple(void) {
    float32_t p_data[] = {1.0f, 2.0f, 3.0f};
    float32_t t_data[] = {1.5f, 2.5f, 2.8f};
    Tensor *pred = create_tensor_1d(p_data, 3, true);
    Tensor *target = create_tensor_1d(t_data, 3, false);

    Tensor *loss = mse_loss(pred, target);
    backward(loss);

    TEST_ASSERT_NOT_NULL(pred->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, -0.3333333f, pred->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -0.3333333f, pred->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.1333333f, pred->grad->data[2]);

    tensor_release(pred);
    tensor_release(target);
    tensor_release(loss);
}

void test_mse_loss_backward_perfect_prediction(void) {
    float32_t data[] = {1.0f, 2.0f, 3.0f};
    Tensor *pred = create_tensor_1d(data, 3, true);
    Tensor *target = create_tensor_1d(data, 3, false);

    Tensor *loss = mse_loss(pred, target);
    backward(loss);

    TEST_ASSERT_NOT_NULL(pred->grad);

    for (uint64_t i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, pred->grad->data[i]);
    }

    tensor_release(pred);
    tensor_release(target);
    tensor_release(loss);
}

void test_mse_loss_backward_target_no_grad(void) {
    float32_t p_data[] = {1.0f, 2.0f};
    float32_t t_data[] = {0.0f, 0.0f};
    Tensor *pred = create_tensor_1d(p_data, 2, true);
    Tensor *target = create_tensor_1d(t_data, 2, true);

    Tensor *loss = mse_loss(pred, target);
    backward(loss);

    TEST_ASSERT_NOT_NULL(pred->grad);
    TEST_ASSERT_NULL(target->grad);

    tensor_release(pred);
    tensor_release(target);
    tensor_release(loss);
}

void test_mse_loss_backward_chain(void) {

    float32_t x_data[] = {1.0f, 2.0f};
    float32_t t_data[] = {4.0f, 6.0f};
    Tensor *x = create_tensor_1d(x_data, 2, true);
    Tensor *two = create_tensor_1d((float32_t[]){2.0f, 2.0f}, 2, false);
    Tensor *target = create_tensor_1d(t_data, 2, false);

    Tensor *x_times_2 = tensor_mul(x, two);
    Tensor *loss = mse_loss(x_times_2, target);
    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, -4.0f, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -4.0f, x->grad->data[1]);

    tensor_release(x);
    tensor_release(two);
    tensor_release(target);
    tensor_release(x_times_2);
    tensor_release(loss);
}

void test_cross_entropy_loss_backward_simple(void) {
    float32_t l_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float32_t t_data[] = {0.0f, 1.0f};
    Tensor *logits = create_tensor_2d(l_data, 2, 2, true);
    Tensor *targets = create_tensor_1d(t_data, 2, false);

    Tensor *loss = cross_entropy_loss(logits, targets);
    backward(loss);

    TEST_ASSERT_NOT_NULL(logits->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, -0.25f, logits->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.25f, logits->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.25f, logits->grad->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -0.25f, logits->grad->data[3]);

    tensor_release(logits);
    tensor_release(targets);
    tensor_release(loss);
}

void test_cross_entropy_loss_backward_target_no_grad(void) {
    float32_t l_data[] = {1.0f, 2.0f, 3.0f};
    float32_t t_data[] = {1.0f};
    Tensor *logits = create_tensor_2d(l_data, 1, 3, true);
    Tensor *targets = create_tensor_1d(t_data, 1, true);

    Tensor *loss = cross_entropy_loss(logits, targets);
    backward(loss);

    TEST_ASSERT_NOT_NULL(logits->grad);
    TEST_ASSERT_NULL(targets->grad);

    tensor_release(logits);
    tensor_release(targets);
    tensor_release(loss);
}

void test_cross_entropy_loss_backward_chain(void) {

    float32_t x_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float32_t bias_data[] = {0.1f, 0.2f, 0.1f, 0.2f};
    float32_t t_data[] = {0.0f, 1.0f};
    Tensor *x = create_tensor_2d(x_data, 2, 2, true);
    Tensor *bias = create_tensor_2d(bias_data, 2, 2, false);
    Tensor *targets = create_tensor_1d(t_data, 2, false);

    Tensor *logits = tensor_add(x, bias);
    Tensor *loss = cross_entropy_loss(logits, targets);
    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_NOT_NULL(x->grad->data);

    tensor_release(x);
    tensor_release(bias);
    tensor_release(targets);
    tensor_release(logits);
    tensor_release(loss);
}

void test_binary_cross_entropy_loss_backward_simple(void) {
    float32_t p_data[] = {0.5f};
    float32_t t_data[] = {1.0f};
    Tensor *pred = create_tensor_1d(p_data, 1, true);
    Tensor *target = create_tensor_1d(t_data, 1, false);

    Tensor *loss = binary_cross_entropy_loss(pred, target);
    backward(loss);

    TEST_ASSERT_NOT_NULL(pred->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, -2.0f, pred->grad->data[0]);

    tensor_release(pred);
    tensor_release(target);
    tensor_release(loss);
}

void test_binary_cross_entropy_loss_backward_batch(void) {
    float32_t p_data[] = {0.5f, 0.5f};
    float32_t t_data[] = {1.0f, 0.0f};
    Tensor *pred = create_tensor_1d(p_data, 2, true);
    Tensor *target = create_tensor_1d(t_data, 2, false);

    Tensor *loss = binary_cross_entropy_loss(pred, target);
    backward(loss);

    TEST_ASSERT_NOT_NULL(pred->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, -1.0f, pred->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, pred->grad->data[1]);

    tensor_release(pred);
    tensor_release(target);
    tensor_release(loss);
}

void test_binary_cross_entropy_loss_backward_target_no_grad(void) {
    float32_t p_data[] = {0.7f, 0.3f};
    float32_t t_data[] = {1.0f, 0.0f};
    Tensor *pred = create_tensor_1d(p_data, 2, true);
    Tensor *target = create_tensor_1d(t_data, 2, true);

    Tensor *loss = binary_cross_entropy_loss(pred, target);
    backward(loss);

    TEST_ASSERT_NOT_NULL(pred->grad);
    TEST_ASSERT_NULL(target->grad);

    tensor_release(pred);
    tensor_release(target);
    tensor_release(loss);
}

void test_binary_cross_entropy_loss_backward_chain(void) {

    float32_t x_data[] = {0.0f, 1.0f};
    float32_t t_data[] = {1.0f, 0.0f};
    Tensor *x = create_tensor_1d(x_data, 2, true);
    Tensor *half = create_tensor_1d((float32_t[]){0.5f, 0.5f}, 2, false);
    Tensor *target = create_tensor_1d(t_data, 2, false);

    Tensor *x_half = tensor_mul(x, half);
    Tensor *pred = tensor_add(x_half, half);
    Tensor *loss = binary_cross_entropy_loss(pred, target);
    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);

    tensor_release(x);
    tensor_release(half);
    tensor_release(target);
    tensor_release(x_half);
    tensor_release(pred);
    tensor_release(loss);
}

void test_loss_chain_with_arithmetic(void) {

    float32_t x_data[] = {1.0f};
    float32_t t_data[] = {5.0f};
    Tensor *x = create_tensor_1d(x_data, 1, true);
    Tensor *two = create_tensor_1d((float32_t[]){2.0f}, 1, false);
    Tensor *one = create_tensor_1d((float32_t[]){1.0f}, 1, false);
    Tensor *target = create_tensor_1d(t_data, 1, false);

    Tensor *x2 = tensor_mul(x, two);
    Tensor *x2_plus_1 = tensor_add(x2, one);
    Tensor *loss = mse_loss(x2_plus_1, target);
    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, -8.0f, x->grad->data[0]);

    tensor_release(x);
    tensor_release(two);
    tensor_release(one);
    tensor_release(target);
    tensor_release(x2);
    tensor_release(x2_plus_1);
    tensor_release(loss);
}

void test_mse_loss_backward_large_values(void) {
    float32_t p_data[] = {1e5f, -1e5f};
    float32_t t_data[] = {1e5f + 1.0f, -1e5f - 1.0f};
    Tensor *pred = create_tensor_1d(p_data, 2, true);
    Tensor *target = create_tensor_1d(t_data, 2, false);

    Tensor *loss = mse_loss(pred, target);
    backward(loss);

    TEST_ASSERT_NOT_NULL(pred->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -1.0f, pred->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f, pred->grad->data[1]);

    tensor_release(pred);
    tensor_release(target);
    tensor_release(loss);
}

void test_mse_loss_backward_broadcast(void) {
    float32_t p_data[] = {1.0f, 2.0f, 3.0f, 4.0f}; // 2x2
    float32_t t_data[] = {1.0f, 2.0f}; // 2x1
    Tensor *pred = create_tensor_2d(p_data, 2, 2, true);
    Tensor *target = create_tensor_2d(t_data, 2, 1, false);

    Tensor *loss = mse_loss(pred, target);
    backward(loss);

    TEST_ASSERT_NOT_NULL(pred->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, pred->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.5f, pred->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.5f, pred->grad->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f, pred->grad->data[3]);

    tensor_release(pred);
    tensor_release(target);
    tensor_release(loss);
}

void test_shared_input_multiple_losses(void) {
    float32_t x_data[] = {1.0f};
    float32_t t1_data[] = {2.0f};
    float32_t t2_data[] = {0.0f};
    
    Tensor *x = create_tensor_1d(x_data, 1, true);
    Tensor *t1 = create_tensor_1d(t1_data, 1, false);
    Tensor *t2 = create_tensor_1d(t2_data, 1, false);
    
    Tensor *l1 = mse_loss(x, t1);
    Tensor *l2 = mse_loss(x, t2);
    
    Tensor *total_loss = tensor_add(l1, l2);
    backward(total_loss);
    
    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, x->grad->data[0]);
    
    tensor_release(x);
    tensor_release(t1);
    tensor_release(t2);
    tensor_release(l1);
    tensor_release(l2);
    tensor_release(total_loss);
}

void test_cross_entropy_loss_backward_large_logits(void) {
    float32_t l_data[] = {100.0f, 100.0f};
    float32_t t_data[] = {0.0f};
    
    Tensor *logits = create_tensor_2d(l_data, 1, 2, true);
    Tensor *targets = create_tensor_1d(t_data, 1, false);
    
    Tensor *loss = cross_entropy_loss(logits, targets);
    backward(loss);
    
    TEST_ASSERT_NOT_NULL(logits->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -0.5f, logits->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.5f, logits->grad->data[1]);
    
    tensor_release(logits);
    tensor_release(targets);
    tensor_release(loss);
}

void test_binary_cross_entropy_loss_edge_predictions(void) {
    float32_t p_data[] = {1e-5f, 1.0f - 1e-5f};
    float32_t t_data[] = {0.0f, 1.0f};
    Tensor *pred = create_tensor_1d(p_data, 2, true);
    Tensor *target = create_tensor_1d(t_data, 2, false);
    
    Tensor *loss = binary_cross_entropy_loss(pred, target);
    backward(loss);
    
    TEST_ASSERT_NOT_NULL(pred->grad);
    TEST_ASSERT_FALSE(isnan(pred->grad->data[0]));
    TEST_ASSERT_FALSE(isnan(pred->grad->data[1]));
    
    tensor_release(pred);
    tensor_release(target);
    tensor_release(loss);
}

void test_binary_cross_entropy_loss_broadcast(void) {
    float32_t p_data[] = {0.5f, 0.5f};
    float32_t t_data[] = {1.0f};
    Tensor *pred = create_tensor_2d(p_data, 2, 1, true);
    Tensor *target = create_tensor_1d(t_data, 1, false);
    
    Tensor *loss = binary_cross_entropy_loss(pred, target);
    backward(loss);
    
    TEST_ASSERT_NOT_NULL(pred->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -1.0f, pred->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, -1.0f, pred->grad->data[1]);
    
    tensor_release(pred);
    tensor_release(target);
    tensor_release(loss);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_mse_loss_backward_simple);
    RUN_TEST(test_mse_loss_backward_perfect_prediction);
    RUN_TEST(test_mse_loss_backward_target_no_grad);
    RUN_TEST(test_mse_loss_backward_chain);
    RUN_TEST(test_cross_entropy_loss_backward_simple);
    RUN_TEST(test_cross_entropy_loss_backward_target_no_grad);
    RUN_TEST(test_cross_entropy_loss_backward_chain);
    RUN_TEST(test_binary_cross_entropy_loss_backward_simple);
    RUN_TEST(test_binary_cross_entropy_loss_backward_batch);
    RUN_TEST(test_binary_cross_entropy_loss_backward_target_no_grad);
    RUN_TEST(test_binary_cross_entropy_loss_backward_chain);
    RUN_TEST(test_loss_chain_with_arithmetic);
    RUN_TEST(test_mse_loss_backward_large_values);
    RUN_TEST(test_mse_loss_backward_broadcast);
    RUN_TEST(test_shared_input_multiple_losses);
    RUN_TEST(test_cross_entropy_loss_backward_large_logits);
    RUN_TEST(test_binary_cross_entropy_loss_edge_predictions);
    RUN_TEST(test_binary_cross_entropy_loss_broadcast);
    return UNITY_END();
}
