#include "../src/ops/losses_backward.h"
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

void test_mse_loss_backward_standard(void) {
    float32_t p_data[] = {1.0f, 2.0f, 3.0f};
    float32_t t_data[] = {1.5f, 2.5f, 2.8f};
    Tensor *pred = create_tensor_1d(p_data, 3);
    Tensor *target = create_tensor_1d(t_data, 3);

    Tensor *grad = mse_loss_backward(pred, target);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, -0.3333333f, grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -0.3333333f, grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.1333333f, grad->data[2]);

    tensor_free(grad);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_backward_size_scaling(void) {
    float32_t p_data[] = {10.0f, 10.0f};
    float32_t t_data[] = {0.0f, 0.0f};
    Tensor *pred = create_tensor_1d(p_data, 2);
    Tensor *target = create_tensor_1d(t_data, 2);

    Tensor *grad = mse_loss_backward(pred, target);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 10.0f, grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 10.0f, grad->data[1]);

    tensor_free(grad);
    tensor_free(pred);
    tensor_free(target);
}

void test_mse_loss_backward_zeros(void) {
    float32_t data[] = {1.0f, 2.0f, -1.0f};
    Tensor *pred = create_tensor_1d(data, 3);
    Tensor *target = create_tensor_1d(data, 3);

    Tensor *grad = mse_loss_backward(pred, target);

    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, grad->data[i]);
    }

    tensor_free(grad);
    tensor_free(pred);
    tensor_free(target);
}

void test_cross_entropy_backward_standard(void) {
    float32_t l_data_in[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float32_t t_data_in[] = {0.0f, 1.0f};

    Tensor *logits = create_tensor_2d(l_data_in, 2, 2);
    Tensor *targets = create_tensor_1d(t_data_in, 2);

    Tensor *grad = cross_entropy_loss_backward(logits, targets);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, -0.25f, grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.25f, grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.25f, grad->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -0.25f, grad->data[3]);

    tensor_free(grad);
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_backward_batch(void) {
    float32_t l_data[] = {100.0f, 0.0f, 0.0f};
    float32_t t_data[] = {0.0f};

    Tensor *logits = create_tensor_2d(l_data, 1, 3);
    Tensor *targets = create_tensor_1d(t_data, 1);

    Tensor *grad = cross_entropy_loss_backward(logits, targets);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.0f, grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.0f, grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.0f, grad->data[2]);

    tensor_free(grad);
    tensor_free(logits);
    tensor_free(targets);
}

void test_cross_entropy_backward_sum_zero(void) {
    float32_t l_data[] = {1.0f, 2.0f, 3.0f};
    float32_t t_data[] = {1.0f};

    Tensor *logits = create_tensor_2d(l_data, 1, 3);
    Tensor *targets = create_tensor_1d(t_data, 1);

    Tensor *grad = cross_entropy_loss_backward(logits, targets);

    float32_t sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        sum += grad->data[i];
    }

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.0f, sum);

    tensor_free(grad);
    tensor_free(logits);
    tensor_free(targets);
}

void test_bce_loss_backward_standard(void) {
    float32_t p_data[] = {0.5f};
    float32_t t_data[] = {1.0f};

    Tensor *pred = create_tensor_1d(p_data, 1);
    Tensor *target = create_tensor_1d(t_data, 1);

    Tensor *grad = binary_cross_entropy_loss_backward(pred, target);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, -2.0f, grad->data[0]);

    tensor_free(grad);
    tensor_free(pred);
    tensor_free(target);
}

void test_bce_loss_backward_batch(void) {
    float32_t p_data[] = {0.5f, 0.5f};
    float32_t t_data[] = {1.0f, 0.0f};

    Tensor *pred = create_tensor_1d(p_data, 2);
    Tensor *target = create_tensor_1d(t_data, 2);

    Tensor *grad = binary_cross_entropy_loss_backward(pred, target);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, -1.0f, grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, grad->data[1]);

    tensor_free(grad);
    tensor_free(pred);
    tensor_free(target);
}

void test_bce_loss_backward_edge_handling(void) {
    float32_t p_data[] = {0.0001f};
    float32_t t_data[] = {0.0f};

    Tensor *pred = create_tensor_1d(p_data, 1);
    Tensor *target = create_tensor_1d(t_data, 1);

    Tensor *grad = binary_cross_entropy_loss_backward(pred, target);

    TEST_ASSERT_FLOAT_WITHIN(0.1f, 1.0f, grad->data[0]);

    tensor_free(grad);
    tensor_free(pred);
    tensor_free(target);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_mse_loss_backward_standard);
    RUN_TEST(test_mse_loss_backward_size_scaling);
    RUN_TEST(test_mse_loss_backward_zeros);
    RUN_TEST(test_cross_entropy_backward_standard);
    RUN_TEST(test_cross_entropy_backward_batch);
    RUN_TEST(test_cross_entropy_backward_sum_zero);
    RUN_TEST(test_bce_loss_backward_standard);
    RUN_TEST(test_bce_loss_backward_batch);
    RUN_TEST(test_bce_loss_backward_edge_handling);
    return UNITY_END();
}
