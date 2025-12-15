#include "../src/losses.h"
#include "../src/tensor.h"
#include "unity.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>

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
    Tensor *loss = mse_loss(pred, target);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, loss->data[0]);
    tensor_free(loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_known_error(void) {
    float32_t p_data[] = {1.0f, 2.0f, 3.0f};
    float32_t t_data[] = {1.5f, 2.5f, 2.8f};
    Tensor *pred = create_tensor_1d(p_data, 3);
    Tensor *target = create_tensor_1d(t_data, 3);
    Tensor *loss_tensor = mse_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.18f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_larger_tensor(void) {
    float32_t p_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float32_t t_data[] = {2.0f, 3.0f, 4.0f, 5.0f};
    Tensor *pred = create_tensor_2d(p_data, 2, 2);
    Tensor *target = create_tensor_2d(t_data, 2, 2);
    Tensor *loss_tensor = mse_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_zeros(void) {
    float32_t data[] = {0.0f, 0.0f, 0.0f};
    Tensor *pred = create_tensor_1d(data, 3);
    Tensor *target = create_tensor_1d(data, 3);
    Tensor *loss_tensor = mse_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_large_values(void) {
    float32_t p_data[] = {1000.0f, 2000.0f};
    float32_t t_data[] = {1001.0f, 1999.0f};
    Tensor *pred = create_tensor_1d(p_data, 2);
    Tensor *target = create_tensor_1d(t_data, 2);
    Tensor *loss_tensor = mse_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_cross_entropy_perfect(void) {
    float32_t l_data[] = {10.0f, -10.0f, -10.0f, 10.0f};
    float32_t t_data[] = {0.0f, 1.0f};
    Tensor *logits = create_tensor_2d(l_data, 2, 2);
    Tensor *targets = create_tensor_1d(t_data, 2);
    Tensor *loss_tensor = cross_entropy_loss(logits, targets);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_TRUE(loss < 1e-4f);
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_uniform(void) {
    float32_t l_data[] = {0.0f, 0.0f, 0.0f};
    float32_t t_data[] = {0.0f};
    Tensor *logits = create_tensor_2d(l_data, 1, 3);
    Tensor *targets = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = cross_entropy_loss(logits, targets);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, logf(3.0f), loss);
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_stability(void) {
    float32_t l_data[] = {1000.0f, 1000.0f, 1000.0f};
    float32_t t_data[] = {0.0f};
    Tensor *logits = create_tensor_2d(l_data, 1, 3);
    Tensor *targets = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = cross_entropy_loss(logits, targets);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, logf(3.0f), loss);
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_batch(void) {
    float32_t l_data[] = {10.0f, -10.0f, -10.0f, 10.0f, 0.0f, 0.0f};
    float32_t t_data[] = {0.0f, 1.0f, 0.0f};
    Tensor *logits = create_tensor_2d(l_data, 3, 2);
    Tensor *targets = create_tensor_1d(t_data, 3);
    Tensor *loss_tensor = cross_entropy_loss(logits, targets);
    float32_t loss = loss_tensor->data[0];
    float32_t expected = (0.0f + 0.0f + logf(2.0f)) / 3.0f;
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, expected, loss);
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_large_negative_logits(void) {
    float32_t l_data[] = {-1000.0f, -1000.0f};
    float32_t t_data[] = {0.0f};
    Tensor *logits = create_tensor_2d(l_data, 1, 2);
    Tensor *targets = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = cross_entropy_loss(logits, targets);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, logf(2.0f), loss);
    tensor_free(logits);
    tensor_free(targets);
}

void test_binary_cross_entropy_perfect(void) {
    float32_t p_data[] = {0.9999f, 0.0001f};
    float32_t t_data[] = {1.0f, 0.0f};
    Tensor *pred = create_tensor_1d(p_data, 2);
    Tensor *target = create_tensor_1d(t_data, 2);
    Tensor *loss = binary_cross_entropy_loss(pred, target);
    TEST_ASSERT_TRUE(loss->data[0] < 1e-3f);
    tensor_free(loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_binary_cross_entropy_worst(void) {
    float32_t p_data[] = {0.0001f};
    float32_t t_data[] = {1.0f};
    Tensor *pred = create_tensor_1d(p_data, 1);
    Tensor *target = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = binary_cross_entropy_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(0.1f, -logf(0.0001f), loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_binary_cross_entropy_clamping(void) {
    float32_t p_data[] = {0.0f, 1.0f};
    float32_t t_data[] = {1.0f, 0.0f};
    Tensor *pred = create_tensor_1d(p_data, 2);
    Tensor *target = create_tensor_1d(t_data, 2);
    Tensor *loss = binary_cross_entropy_loss(pred, target);
    TEST_ASSERT_FALSE(isinf(loss->data[0]));
    TEST_ASSERT_FALSE(isnan(loss->data[0]));
    TEST_ASSERT_TRUE(loss->data[0] > 10.0f);
    tensor_free(loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_binary_cross_entropy_50_50(void) {
    float32_t p_data[] = {0.5f, 0.5f};
    float32_t t_data[] = {1.0f, 0.0f};
    Tensor *pred = create_tensor_1d(p_data, 2);
    Tensor *target = create_tensor_1d(t_data, 2);
    Tensor *loss_tensor = binary_cross_entropy_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, -logf(0.5f), loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_binary_cross_entropy_mixed(void) {
    float32_t p_data[] = {0.8f, 0.2f, 0.6f};
    float32_t t_data[] = {1.0f, 0.0f, 1.0f};
    Tensor *pred = create_tensor_1d(p_data, 3);
    Tensor *target = create_tensor_1d(t_data, 3);
    Tensor *loss = binary_cross_entropy_loss(pred, target);

    float32_t expected = -(logf(0.8f) + logf(0.8f) + logf(0.6f)) / 3.0f;
    TEST_ASSERT_FLOAT_WITHIN(1e-5, expected, loss->data[0]);

    tensor_free(loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_small_diff(void) {
    float32_t p_data[] = {1.000001f};
    float32_t t_data[] = {1.0f};
    Tensor *pred = create_tensor_1d(p_data, 1);
    Tensor *target = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = mse_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-13, 1e-12, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_cross_entropy_logits_max_shift_invariance(void) {
    float32_t l_data1[] = {1.0f, 2.0f, 3.0f};
    float32_t l_data2[] = {101.0f, 102.0f, 103.0f};
    float32_t t_data[] = {2.0f};

    Tensor *logits1 = create_tensor_2d(l_data1, 1, 3);
    Tensor *logits2 = create_tensor_2d(l_data2, 1, 3);
    Tensor *targets = create_tensor_1d(t_data, 1);

    Tensor *loss_tensor1 = cross_entropy_loss(logits1, targets);
    float32_t loss1 = loss_tensor1->data[0];
    tensor_free(loss_tensor1);

    Tensor *loss_tensor2 = cross_entropy_loss(logits2, targets);
    float32_t loss2 = loss_tensor2->data[0];
    tensor_free(loss_tensor2);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, loss1, loss2);

    tensor_free(logits1);
    tensor_free(logits2);
    tensor_free(targets);
}

void test_cross_entropy_single_class(void) {
    float32_t l_data[] = {10.0f, -5.0f};
    float32_t t_data[] = {0.0f, 0.0f};

    Tensor *logits = create_tensor_2d(l_data, 2, 1);
    Tensor *targets = create_tensor_1d(t_data, 2);

    Tensor *loss = cross_entropy_loss(logits, targets);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, loss->data[0]);

    tensor_free(loss);
    tensor_free(logits);
    tensor_free(targets);
}

void test_binary_cross_entropy_boundary_handling(void) {
    float32_t p_data[] = {-0.5f, 0.0f, 1.0f, 1.5f};
    float32_t t_data[] = {0.0f, 0.0f, 1.0f, 1.0f};

    Tensor *pred = create_tensor_1d(p_data, 4);
    Tensor *target = create_tensor_1d(t_data, 4);

    Tensor *loss = binary_cross_entropy_loss(pred, target);

    TEST_ASSERT_FALSE(isnan(loss->data[0]));
    TEST_ASSERT_FALSE(isinf(loss->data[0]));

    tensor_free(loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_scalar(void) {
    float32_t p_data[] = {2.5f};
    float32_t t_data[] = {2.0f};
    Tensor *pred = create_tensor_1d(p_data, 1);
    Tensor *target = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = mse_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.25f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_3d_tensor(void) {
    float32_t p_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float32_t t_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    uint64_t shape[] = {2, 2, 2};
    Tensor *pred = tensor_create(p_data, shape, 3, false);
    Tensor *target = tensor_create(t_data, shape, 3, false);
    Tensor *loss = mse_loss(pred, target);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, loss->data[0]);
    tensor_free(loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_4d_tensor(void) {
    float32_t p_data[16];
    float32_t t_data[16];
    for (int i = 0; i < 16; i++) {
        p_data[i] = (float)i;
        t_data[i] = (float)i + 1.0f;
    }
    uint64_t shape[] = {2, 2, 2, 2};
    Tensor *pred = tensor_create(p_data, shape, 4, false);
    Tensor *target = tensor_create(t_data, shape, 4, false);
    Tensor *loss_tensor = mse_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_fractional_values(void) {
    float32_t p_data[] = {0.1f, 0.2f};
    float32_t t_data[] = {0.2f, 0.3f};
    Tensor *pred = create_tensor_1d(p_data, 2);
    Tensor *target = create_tensor_1d(t_data, 2);
    Tensor *loss_tensor = mse_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.01f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_alternating_signs(void) {
    float32_t p_data[] = {1.0f, -1.0f, 1.0f, -1.0f};
    float32_t t_data[] = {-1.0f, 1.0f, -1.0f, 1.0f};
    Tensor *pred = create_tensor_1d(p_data, 4);
    Tensor *target = create_tensor_1d(t_data, 4);
    Tensor *loss_tensor = mse_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 4.0f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_all_negative(void) {
    float32_t p_data[] = {-2.0f, -5.0f};
    float32_t t_data[] = {-2.0f, -5.0f};
    Tensor *pred = create_tensor_1d(p_data, 2);
    Tensor *target = create_tensor_1d(t_data, 2);
    Tensor *loss = mse_loss(pred, target);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, loss->data[0]);
    tensor_free(loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_increasing_diff(void) {
    float32_t p_data[] = {1.0f, 2.0f, 3.0f};
    float32_t t_data[] = {2.0f, 4.0f, 6.0f};
    Tensor *pred = create_tensor_1d(p_data, 3);
    Tensor *target = create_tensor_1d(t_data, 3);
    Tensor *loss_tensor = mse_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 4.66666f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_large_diff(void) {
    float32_t p_data[] = {0.0f};
    float32_t t_data[] = {1000.0f};
    Tensor *pred = create_tensor_1d(p_data, 1);
    Tensor *target = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = mse_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1000000.0f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_sub_epsilon(void) {
    float32_t p_data[] = {1.0f};
    float32_t t_data[] = {1.0f + FLT_EPSILON / 2.0f};
    Tensor *pred = create_tensor_1d(p_data, 1);
    Tensor *target = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = mse_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_TRUE(loss < 1e-10f);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_identity(void) {
    float32_t p_data[] = {123.456f, 789.012f};
    Tensor *pred = create_tensor_1d(p_data, 2);
    Tensor *target = create_tensor_1d(p_data, 2);
    Tensor *loss = mse_loss(pred, target);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, loss->data[0]);
    tensor_free(loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_cross_entropy_single_item_batch(void) {
    float32_t l_data[] = {2.0f, 1.0f, 0.1f};
    float32_t t_data[] = {0.0f};
    Tensor *logits = create_tensor_2d(l_data, 1, 3);
    Tensor *targets = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = cross_entropy_loss(logits, targets);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(2e-4, 0.4170f, loss);
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_large_batch(void) {
    uint64_t batch_size = 10;
    uint64_t num_classes = 2;
    float32_t *l_data = (float32_t *)malloc(batch_size * num_classes * sizeof(float32_t));
    float32_t *t_data = (float32_t *)malloc(batch_size * sizeof(float32_t));

    for (uint64_t i = 0; i < batch_size; i++) {
        l_data[i * 2] = 10.0f;
        l_data[i * 2 + 1] = -10.0f;
        t_data[i] = 0.0f;
    }

    Tensor *logits = create_tensor_2d(l_data, batch_size, num_classes);
    Tensor *targets = create_tensor_1d(t_data, batch_size);
    Tensor *loss_tensor = cross_entropy_loss(logits, targets);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);

    TEST_ASSERT_TRUE(loss < 1e-4f);

    tensor_free(logits);
    tensor_free(targets);
    free(l_data);
    free(t_data);
}

void test_cross_entropy_high_dimension_classes(void) {
    uint64_t num_classes = 100;
    float32_t *l_data = (float32_t *)calloc(num_classes, sizeof(float32_t));
    l_data[50] = 100.0f;
    float32_t t_data[] = {50.0f};

    Tensor *logits = create_tensor_2d(l_data, 1, num_classes);
    Tensor *targets = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = cross_entropy_loss(logits, targets);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);

    TEST_ASSERT_TRUE(loss < 1e-4f);

    tensor_free(logits);
    tensor_free(targets);
    free(l_data);
}

void test_cross_entropy_target_zero(void) {
    float32_t l_data[] = {0.5f, 0.2f, 0.1f};
    float32_t t_data[] = {0.0f};
    Tensor *logits = create_tensor_2d(l_data, 1, 3);
    Tensor *targets = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = cross_entropy_loss(logits, targets);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FALSE(isnan(loss));
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_target_last(void) {
    float32_t l_data[] = {0.1f, 0.2f, 0.5f};
    float32_t t_data[] = {2.0f};
    Tensor *logits = create_tensor_2d(l_data, 1, 3);
    Tensor *targets = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = cross_entropy_loss(logits, targets);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FALSE(isnan(loss));
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_logits_all_equal(void) {
    float32_t l_data[] = {5.0f, 5.0f, 5.0f, 5.0f};
    float32_t t_data[] = {1.0f};
    Tensor *logits = create_tensor_2d(l_data, 1, 4);
    Tensor *targets = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = cross_entropy_loss(logits, targets);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.38629f, loss);
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_distinct_logits(void) {
    float32_t l_data[] = {1.0f, 2.0f, 3.0f};
    float32_t t_data[] = {2.0f};
    Tensor *logits = create_tensor_2d(l_data, 1, 3);
    Tensor *targets = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = cross_entropy_loss(logits, targets);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_TRUE(loss < 1.0f);
    TEST_ASSERT_TRUE(loss >= 0.0f);
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_very_large_logits(void) {
    float32_t l_data[] = {10000.0f, 0.0f};
    float32_t t_data[] = {0.0f};
    Tensor *logits = create_tensor_2d(l_data, 1, 2);
    Tensor *targets = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = cross_entropy_loss(logits, targets);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, loss);
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_very_small_logits(void) {
    float32_t l_data[] = {-10000.0f, -10000.0f};
    float32_t t_data[] = {0.0f};
    Tensor *logits = create_tensor_2d(l_data, 1, 2);
    Tensor *targets = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = cross_entropy_loss(logits, targets);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-3, 0.69314f, loss);
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_mixed_logits(void) {
    float32_t l_data[] = {100.0f, -100.0f, 50.0f};
    float32_t t_data[] = {0.0f};
    Tensor *logits = create_tensor_2d(l_data, 1, 3);
    Tensor *targets = create_tensor_1d(t_data, 1);
    Tensor *loss = cross_entropy_loss(logits, targets);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, loss->data[0]);
    tensor_free(loss);
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_soft_max_dominance(void) {
    float32_t l_data[] = {10.0f, 0.0f, 0.0f};
    float32_t t_data[] = {0.0f};
    Tensor *logits = create_tensor_2d(l_data, 1, 3);
    Tensor *targets = create_tensor_1d(t_data, 1);
    Tensor *loss = cross_entropy_loss(logits, targets);
    TEST_ASSERT_TRUE(loss->data[0] < 1e-4f);
    tensor_free(loss);
    tensor_free(logits);
    tensor_free(targets);
}

void test_bce_loss_scalar(void) {
    float32_t p_data[] = {0.8f};
    float32_t t_data[] = {1.0f};
    Tensor *pred = create_tensor_1d(p_data, 1);
    Tensor *target = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = binary_cross_entropy_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.22314f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_bce_loss_target_0_pred_0(void) {
    float32_t p_data[] = {0.0f};
    float32_t t_data[] = {0.0f};
    Tensor *pred = create_tensor_1d(p_data, 1);
    Tensor *target = create_tensor_1d(t_data, 1);
    Tensor *loss = binary_cross_entropy_loss(pred, target);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, loss->data[0]);
    tensor_free(loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_bce_loss_target_1_pred_1(void) {
    float32_t p_data[] = {1.0f};
    float32_t t_data[] = {1.0f};
    Tensor *pred = create_tensor_1d(p_data, 1);
    Tensor *target = create_tensor_1d(t_data, 1);
    Tensor *loss = binary_cross_entropy_loss(pred, target);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, loss->data[0]);
    tensor_free(loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_bce_loss_pred_epsilon(void) {
    float32_t p_data[] = {FLT_EPSILON};
    float32_t t_data[] = {0.0f};
    Tensor *pred = create_tensor_1d(p_data, 1);
    Tensor *target = create_tensor_1d(t_data, 1);
    Tensor *loss = binary_cross_entropy_loss(pred, target);
    TEST_ASSERT_TRUE(loss->data[0] < 1e-6f);
    tensor_free(loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_bce_loss_pred_one_minus_epsilon(void) {
    float32_t p_data[] = {1.0f - FLT_EPSILON};
    float32_t t_data[] = {1.0f};
    Tensor *pred = create_tensor_1d(p_data, 1);
    Tensor *target = create_tensor_1d(t_data, 1);
    Tensor *loss = binary_cross_entropy_loss(pred, target);
    TEST_ASSERT_TRUE(loss->data[0] < 1e-6f);
    tensor_free(loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_bce_loss_soft_labels(void) {
    float32_t p_data[] = {0.8f};
    float32_t t_data[] = {0.5f};
    Tensor *pred = create_tensor_1d(p_data, 1);
    Tensor *target = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = binary_cross_entropy_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.91629f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_bce_loss_soft_labels_pred_match(void) {
    float32_t p_data[] = {0.5f};
    float32_t t_data[] = {0.5f};
    Tensor *pred = create_tensor_1d(p_data, 1);
    Tensor *target = create_tensor_1d(t_data, 1);
    Tensor *loss_tensor = binary_cross_entropy_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.69314f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_bce_loss_unbalanced_batch(void) {
    float32_t p_data[] = {0.9f, 0.1f, 0.5f};
    float32_t t_data[] = {1.0f, 0.0f, 1.0f};
    Tensor *pred = create_tensor_1d(p_data, 3);
    Tensor *target = create_tensor_1d(t_data, 3);
    Tensor *loss_tensor = binary_cross_entropy_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.3012f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_bce_loss_alternating(void) {
    float32_t p_data[] = {0.1f, 0.9f, 0.1f, 0.9f};
    float32_t t_data[] = {0.0f, 1.0f, 0.0f, 1.0f};
    Tensor *pred = create_tensor_1d(p_data, 4);
    Tensor *target = create_tensor_1d(t_data, 4);
    Tensor *loss_tensor = binary_cross_entropy_loss(pred, target);
    float32_t loss = loss_tensor->data[0];
    tensor_free(loss_tensor);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.10536f, loss);
    tensor_free(pred);
    tensor_free(target);
}

void test_bce_loss_large_batch_zeros(void) {
    uint64_t size = 100;
    float32_t *p_data = (float32_t *)calloc(size, sizeof(float32_t));
    float32_t *t_data = (float32_t *)calloc(size, sizeof(float32_t));
    // predictions 0.0, targets 0.0 -> loss 0
    Tensor *pred = create_tensor_1d(p_data, size);
    Tensor *target = create_tensor_1d(t_data, size);
    Tensor *loss = binary_cross_entropy_loss(pred, target);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, loss->data[0]);
    tensor_free(loss);
    tensor_free(pred);
    tensor_free(target);
    free(p_data);
    free(t_data);
}

void test_bce_loss_large_batch_ones(void) {
    uint64_t size = 100;
    float32_t *p_data = (float32_t *)malloc(size * sizeof(float32_t));
    float32_t *t_data = (float32_t *)malloc(size * sizeof(float32_t));
    for (uint64_t i = 0; i < size; i++) {
        p_data[i] = 1.0f;
        t_data[i] = 1.0f;
    }
    Tensor *pred = create_tensor_1d(p_data, size);
    Tensor *target = create_tensor_1d(t_data, size);
    Tensor *loss = binary_cross_entropy_loss(pred, target);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, loss->data[0]);
    tensor_free(loss);
    tensor_free(pred);
    tensor_free(target);
    free(p_data);
    free(t_data);
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
    RUN_TEST(test_mse_loss_small_diff);
    RUN_TEST(test_cross_entropy_logits_max_shift_invariance);
    RUN_TEST(test_cross_entropy_single_class);
    RUN_TEST(test_binary_cross_entropy_boundary_handling);
    RUN_TEST(test_mse_loss_scalar);
    RUN_TEST(test_mse_loss_3d_tensor);
    RUN_TEST(test_mse_loss_4d_tensor);
    RUN_TEST(test_mse_loss_fractional_values);
    RUN_TEST(test_mse_loss_alternating_signs);
    RUN_TEST(test_mse_loss_all_negative);
    RUN_TEST(test_mse_loss_increasing_diff);
    RUN_TEST(test_mse_loss_large_diff);
    RUN_TEST(test_mse_loss_sub_epsilon);
    RUN_TEST(test_mse_loss_identity);
    RUN_TEST(test_cross_entropy_single_item_batch);
    RUN_TEST(test_cross_entropy_large_batch);
    RUN_TEST(test_cross_entropy_high_dimension_classes);
    RUN_TEST(test_cross_entropy_target_zero);
    RUN_TEST(test_cross_entropy_target_last);
    RUN_TEST(test_cross_entropy_logits_all_equal);
    RUN_TEST(test_cross_entropy_distinct_logits);
    RUN_TEST(test_cross_entropy_very_large_logits);
    RUN_TEST(test_cross_entropy_very_small_logits);
    RUN_TEST(test_cross_entropy_mixed_logits);
    RUN_TEST(test_cross_entropy_soft_max_dominance);
    RUN_TEST(test_bce_loss_scalar);
    RUN_TEST(test_bce_loss_target_0_pred_0);
    RUN_TEST(test_bce_loss_target_1_pred_1);
    RUN_TEST(test_bce_loss_pred_epsilon);
    RUN_TEST(test_bce_loss_pred_one_minus_epsilon);
    RUN_TEST(test_bce_loss_soft_labels);
    RUN_TEST(test_bce_loss_soft_labels_pred_match);
    RUN_TEST(test_bce_loss_unbalanced_batch);
    RUN_TEST(test_bce_loss_alternating);
    RUN_TEST(test_bce_loss_large_batch_zeros);
    RUN_TEST(test_bce_loss_large_batch_ones);
    return UNITY_END();
}
