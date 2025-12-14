#include "../src/losses.h"
#include "../src/tensor.h"
#include "unity.h"
#include <float.h>
#include <math.h>

void setUp(void) {}
void tearDown(void) {}

static Tensor *create_tensor_1d(float32_t *data, uint64_t size) {
    uint64_t shape[] = {size};
    return tensor_create(data, shape, 1, false);
}

static Tensor *create_tensor_2d(float32_t *data, uint64_t rows, uint64_t cols) {
    uint64_t shape[] = {rows, cols};
    return tensor_create(data, shape, 2, false);
}

void test_mse_loss_perfect_prediction(void) {
    float32_t data[] = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f};
    Tensor *pred = create_tensor_1d(data, 5);
    Tensor *target = create_tensor_1d(data, 5);
    float32_t loss = mse_loss(pred, target);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_known_error(void) {
    float32_t p_data[] = {1.0f, 2.0f, 3.0f};
    float32_t t_data[] = {1.5f, 2.5f, 2.8f};
    Tensor *pred = create_tensor_1d(p_data, 3);
    Tensor *target = create_tensor_1d(t_data, 3);
    float32_t loss = mse_loss(pred, target);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.18f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_larger_tensor(void) {
    float32_t p_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float32_t t_data[] = {2.0f, 3.0f, 4.0f, 5.0f};
    Tensor *pred = create_tensor_2d(p_data, 2, 2);
    Tensor *target = create_tensor_2d(t_data, 2, 2);
    float32_t loss = mse_loss(pred, target);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_zeros(void) {
    float32_t data[] = {0.0f, 0.0f, 0.0f};
    Tensor *pred = create_tensor_1d(data, 3);
    Tensor *target = create_tensor_1d(data, 3);
    float32_t loss = mse_loss(pred, target);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_large_values(void) {
    float32_t p_data[] = {1000.0f, 2000.0f};
    float32_t t_data[] = {1001.0f, 1999.0f};
    Tensor *pred = create_tensor_1d(p_data, 2);
    Tensor *target = create_tensor_1d(t_data, 2);
    float32_t loss = mse_loss(pred, target);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_cross_entropy_perfect(void) {
    float32_t l_data[] = {10.0f, -10.0f, -10.0f, 10.0f};
    float32_t t_data[] = {0.0f, 1.0f};
    Tensor *logits = create_tensor_2d(l_data, 2, 2);
    Tensor *targets = create_tensor_1d(t_data, 2);
    float32_t loss = cross_entropy_loss(logits, targets);
    TEST_ASSERT_TRUE(loss < 1e-4f);
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_uniform(void) {
    float32_t l_data[] = {0.0f, 0.0f, 0.0f};
    float32_t t_data[] = {0.0f};
    Tensor *logits = create_tensor_2d(l_data, 1, 3);
    Tensor *targets = create_tensor_1d(t_data, 1);
    float32_t loss = cross_entropy_loss(logits, targets);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, logf(3.0f), loss);
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_stability(void) {
    float32_t l_data[] = {1000.0f, 1000.0f, 1000.0f};
    float32_t t_data[] = {0.0f};
    Tensor *logits = create_tensor_2d(l_data, 1, 3);
    Tensor *targets = create_tensor_1d(t_data, 1);
    float32_t loss = cross_entropy_loss(logits, targets);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, logf(3.0f), loss);
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_batch(void) {
    float32_t l_data[] = {10.0f, -10.0f, -10.0f, 10.0f, 0.0f, 0.0f};
    float32_t t_data[] = {0.0f, 1.0f, 0.0f};
    Tensor *logits = create_tensor_2d(l_data, 3, 2);
    Tensor *targets = create_tensor_1d(t_data, 3);
    float32_t loss = cross_entropy_loss(logits, targets);
    float32_t expected = (0.0f + 0.0f + logf(2.0f)) / 3.0f;
    TEST_ASSERT_FLOAT_WITHIN(1e-4, expected, loss);
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_large_negative_logits(void) {
    float32_t l_data[] = {-1000.0f, -1000.0f};
    float32_t t_data[] = {0.0f};
    Tensor *logits = create_tensor_2d(l_data, 1, 2);
    Tensor *targets = create_tensor_1d(t_data, 1);
    float32_t loss = cross_entropy_loss(logits, targets);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, logf(2.0f), loss);
    tensor_free(logits);
    tensor_free(targets);
}

void test_binary_cross_entropy_perfect(void) {
    float32_t p_data[] = {0.9999f, 0.0001f};
    float32_t t_data[] = {1.0f, 0.0f};
    Tensor *pred = create_tensor_1d(p_data, 2);
    Tensor *target = create_tensor_1d(t_data, 2);
    float32_t loss = binary_cross_entropy_loss(pred, target);
    TEST_ASSERT_TRUE(loss < 1e-3f);
    tensor_free(pred);
    tensor_free(target);
}

void test_binary_cross_entropy_worst(void) {
    float32_t p_data[] = {0.0001f};
    float32_t t_data[] = {1.0f};
    Tensor *pred = create_tensor_1d(p_data, 1);
    Tensor *target = create_tensor_1d(t_data, 1);
    float32_t loss = binary_cross_entropy_loss(pred, target);
    TEST_ASSERT_FLOAT_WITHIN(0.1f, -logf(0.0001f), loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_binary_cross_entropy_clamping(void) {
    float32_t p_data[] = {0.0f, 1.0f};
    float32_t t_data[] = {1.0f, 0.0f};
    Tensor *pred = create_tensor_1d(p_data, 2);
    Tensor *target = create_tensor_1d(t_data, 2);
    float32_t loss = binary_cross_entropy_loss(pred, target);
    TEST_ASSERT_FALSE(isinf(loss));
    TEST_ASSERT_FALSE(isnan(loss));
    TEST_ASSERT_TRUE(loss > 10.0f);
    tensor_free(pred);
    tensor_free(target);
}

void test_binary_cross_entropy_50_50(void) {
    float32_t p_data[] = {0.5f, 0.5f};
    float32_t t_data[] = {1.0f, 0.0f};
    Tensor *pred = create_tensor_1d(p_data, 2);
    Tensor *target = create_tensor_1d(t_data, 2);
    float32_t loss = binary_cross_entropy_loss(pred, target);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, -logf(0.5f), loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_binary_cross_entropy_mixed(void) {
    float32_t p_data[] = {0.8f, 0.2f, 0.6f};
    float32_t t_data[] = {1.0f, 0.0f, 1.0f};
    Tensor *pred = create_tensor_1d(p_data, 3);
    Tensor *target = create_tensor_1d(t_data, 3);
    float32_t loss = binary_cross_entropy_loss(pred, target);

    float32_t expected = -(logf(0.8f) + logf(0.8f) + logf(0.6f)) / 3.0f;
    TEST_ASSERT_FLOAT_WITHIN(1e-5, expected, loss);

    tensor_free(pred);
    tensor_free(target);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_mse_loss_perfect_prediction);
    RUN_TEST(test_mse_loss_known_error);
    RUN_TEST(test_mse_loss_larger_tensor);
    RUN_TEST(test_mse_loss_zeros);
    RUN_TEST(test_mse_loss_large_values);
    RUN_TEST(test_cross_entropy_perfect);
    RUN_TEST(test_cross_entropy_uniform);
    RUN_TEST(test_cross_entropy_stability);
    RUN_TEST(test_cross_entropy_batch);
    RUN_TEST(test_cross_entropy_large_negative_logits);
    RUN_TEST(test_binary_cross_entropy_perfect);
    RUN_TEST(test_binary_cross_entropy_worst);
    RUN_TEST(test_binary_cross_entropy_clamping);
    RUN_TEST(test_binary_cross_entropy_50_50);
    RUN_TEST(test_binary_cross_entropy_mixed);
    return UNITY_END();
}
