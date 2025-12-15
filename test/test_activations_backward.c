#include "../src/ops/activations.h"
#include "../src/ops/activations_backward.h"
#include "../src/tensor.h"
#include "unity.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

static Tensor *create_tensor_from_data(float32_t *data, uint64_t size) {
    uint64_t shape[] = {size};
    return tensor_create(data, shape, 1, false);
}

void test_sigmoid_backward_standard_values(void) {
    float32_t data[] = {0.0f, 1.0f, -1.0f};
    Tensor *t = create_tensor_from_data(data, 3);
    Tensor *grad = tensor_sigmoid_backward(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.25f, grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.1966119f, grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.1966119f, grad->data[2]);

    tensor_free(t);
    tensor_free(grad);
}

void test_relu_backward_standard_values(void) {
    float32_t data[] = {-5.0f, -0.1f, 0.0f, 0.1f, 5.0f};
    Tensor *t = create_tensor_from_data(data, 5);
    Tensor *grad = tensor_relu_backward(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, grad->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, grad->data[3]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, grad->data[4]);

    tensor_free(t);
    tensor_free(grad);
}

void test_tanh_backward_standard_values(void) {
    float32_t data[] = {0.0f, 1.0f, -1.0f};
    Tensor *t = create_tensor_from_data(data, 3);
    Tensor *grad = tensor_tanh_backward(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, grad->data[0]);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.4199743f, grad->data[1]);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.4199743f, grad->data[2]);

    tensor_free(t);
    tensor_free(grad);
}

void test_gelu_backward_standard_values(void) {
    float32_t data[] = {0.0f, 1.0f, -1.0f};
    Tensor *t = create_tensor_from_data(data, 3);
    Tensor *grad = tensor_gelu_backward(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.5f, grad->data[0]);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 1.08296f, grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, -0.08296f, grad->data[2]);

    tensor_free(t);
    tensor_free(grad);
}

void test_softmax_backward_diagonal(void) {
    float32_t data[] = {0.0f, 0.0f}; // softmax -> [0.5, 0.5]
    uint64_t shape[] = {2};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *grad = tensor_softmax_backward(t, 0);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.25f, grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.25f, grad->data[1]);

    tensor_free(t);
    tensor_free(grad);
}

void test_sigmoid_backward_stability(void) {
    float32_t data[] = {100.0f, -100.0f};
    Tensor *t = create_tensor_from_data(data, 2);
    Tensor *grad = tensor_sigmoid_backward(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, grad->data[1]);

    tensor_free(t);
    tensor_free(grad);
}

void test_tanh_backward_stability(void) {
    float32_t data[] = {50.0f, -50.0f};
    Tensor *t = create_tensor_from_data(data, 2);
    Tensor *grad = tensor_tanh_backward(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, grad->data[1]);

    tensor_free(t);
    tensor_free(grad);
}

void test_sigmoid_backward_edge_cases(void) {
    float32_t data[] = {0.0f, 100.0f, -100.0f};
    Tensor *t = create_tensor_from_data(data, 3);
    Tensor *grad = tensor_sigmoid_backward(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.25f, grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, grad->data[2]);

    tensor_free(t);
    tensor_free(grad);
}

void test_relu_backward_edge_cases(void) {
    float32_t data[] = {0.0f, 10.0f, -10.0f};
    Tensor *t = create_tensor_from_data(data, 3);
    Tensor *grad = tensor_relu_backward(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, grad->data[2]);

    tensor_free(t);
    tensor_free(grad);
}

void test_tanh_backward_edge_cases(void) {
    float32_t data[] = {0.0f, 50.0f, -50.0f};
    Tensor *t = create_tensor_from_data(data, 3);
    Tensor *grad = tensor_tanh_backward(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, grad->data[2]);

    tensor_free(t);
    tensor_free(grad);
}

void test_gelu_backward_edge_cases(void) {
    float32_t data[] = {0.0f, 1.0f, -1.0f};
    Tensor *t = create_tensor_from_data(data, 3);
    Tensor *grad = tensor_gelu_backward(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.5f, grad->data[0]);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 1.08296f, grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, -0.08296f, grad->data[2]);

    tensor_free(t);
    tensor_free(grad);
}

void test_softmax_backward_shapes(void) {
    float32_t data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint64_t shape[] = {2, 2};
    Tensor *t = tensor_create(data, shape, 2, false);

    Tensor *grad = tensor_softmax_backward(t, 1);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.25f, grad->data[i]);
    }

    tensor_free(grad);

    grad = tensor_softmax_backward(t, 0);
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.25f, grad->data[i]);
    }

    tensor_free(t);
    tensor_free(grad);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_sigmoid_backward_standard_values);
    RUN_TEST(test_relu_backward_standard_values);
    RUN_TEST(test_tanh_backward_standard_values);
    RUN_TEST(test_gelu_backward_standard_values);
    RUN_TEST(test_softmax_backward_diagonal);
    RUN_TEST(test_sigmoid_backward_stability);
    RUN_TEST(test_sigmoid_backward_edge_cases);
    RUN_TEST(test_relu_backward_edge_cases);
    RUN_TEST(test_tanh_backward_edge_cases);
    RUN_TEST(test_gelu_backward_edge_cases);
    RUN_TEST(test_softmax_backward_shapes);
    RUN_TEST(test_tanh_backward_stability);
    return UNITY_END();
}
