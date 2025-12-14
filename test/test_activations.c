#include "../src/activations.h"
#include "../src/tensor.h"
#include "unity.h"
#include <float.h>
#include <math.h>
#include <stdint.h>

void setUp(void) {}
void tearDown(void) {}

static Tensor *create_tensor_from_data(float32_t *data, uint64_t size) {
    uint64_t shape[] = {size};
    return tensor_create(data, shape, 1, false);
}

void test_sigmoid_standard_values(void) {
    float32_t data[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f};
    Tensor *t = create_tensor_from_data(data, 5);
    Tensor *out = tensor_sigmoid(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.5f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.7310586f, out->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.2689414f, out->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.6224593f, out->data[3]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.3775407f, out->data[4]);

    tensor_free(t);
    tensor_free(out);
}

void test_sigmoid_stability_large_positive(void) {
    float32_t data[] = {100.0f, 500.0f, 1000.0f, 1e6f};
    Tensor *t = create_tensor_from_data(data, 4);
    Tensor *out = tensor_sigmoid(t);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, out->data[i]);
    }

    tensor_free(t);
    tensor_free(out);
}

void test_sigmoid_stability_large_negative(void) {
    float32_t data[] = {-100.0f, -500.0f, -1000.0f, -1e6f};
    Tensor *t = create_tensor_from_data(data, 4);
    Tensor *out = tensor_sigmoid(t);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, out->data[i]);
    }

    tensor_free(t);
    tensor_free(out);
}

void test_sigmoid_nan_inf(void) {
    float32_t data[] = {NAN, INFINITY, -INFINITY};
    Tensor *t = create_tensor_from_data(data, 3);
    Tensor *out = tensor_sigmoid(t);

    TEST_ASSERT_TRUE(isnan(out->data[0]));
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, out->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, out->data[2]);

    tensor_free(t);
    tensor_free(out);
}

void test_sigmoid_tiny_values(void) {
    float32_t data[] = {1e-10f, -1e-10f};
    Tensor *t = create_tensor_from_data(data, 2);
    Tensor *out = tensor_sigmoid(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.5f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.5f, out->data[1]);

    tensor_free(t);
    tensor_free(out);
}

void test_relu_standard_values(void) {
    float32_t data[] = {-5.0f, -0.1f, 0.0f, 0.1f, 5.0f};
    Tensor *t = create_tensor_from_data(data, 5);
    Tensor *out = tensor_relu(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, out->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, out->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.1f, out->data[3]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 5.0f, out->data[4]);

    tensor_free(t);
    tensor_free(out);
}

void test_relu_stability(void) {
    float32_t data[] = {-1e6f, 1e6f};
    Tensor *t = create_tensor_from_data(data, 2);
    Tensor *out = tensor_relu(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1e6f, out->data[1]);

    tensor_free(t);
    tensor_free(out);
}

void test_relu_nan_inf(void) {
    float32_t data[] = {NAN, INFINITY, -INFINITY};
    Tensor *t = create_tensor_from_data(data, 3);
    Tensor *out = tensor_relu(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, out->data[0]);
    TEST_ASSERT_TRUE(isinf(out->data[1]) && out->data[1] > 0);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, out->data[2]);

    tensor_free(t);
    tensor_free(out);
}

void test_relu_mixed_sign(void) {
    float32_t data[] = {-1.0f, 1.0f, -2.0f, 2.0f};
    Tensor *t = create_tensor_from_data(data, 4);
    Tensor *out = tensor_relu(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, out->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, out->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 2.0f, out->data[3]);

    tensor_free(t);
    tensor_free(out);
}

void test_tanh_standard_values(void) {
    float32_t data[] = {0.0f, 1.0f, -1.0f};
    Tensor *t = create_tensor_from_data(data, 3);
    Tensor *out = tensor_tanh(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.7615942f, out->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, -0.7615942f, out->data[2]);

    tensor_free(t);
    tensor_free(out);
}

void test_tanh_stability(void) {
    float32_t data[] = {20.0f, -20.0f, 1000.0f, -1000.0f};
    Tensor *t = create_tensor_from_data(data, 4);
    Tensor *out = tensor_tanh(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, -1.0f, out->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, out->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, -1.0f, out->data[3]);

    tensor_free(t);
    tensor_free(out);
}

void test_tanh_nan_inf(void) {
    float32_t data[] = {NAN, INFINITY, -INFINITY};
    Tensor *t = create_tensor_from_data(data, 3);
    Tensor *out = tensor_tanh(t);

    TEST_ASSERT_TRUE(isnan(out->data[0]));
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, out->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, -1.0f, out->data[2]);

    tensor_free(t);
    tensor_free(out);
}

void test_tanh_small_epsilon(void) {
    float32_t data[] = {1e-5f, -1e-5f};
    Tensor *t = create_tensor_from_data(data, 2);
    Tensor *out = tensor_tanh(t);
    TEST_ASSERT_FLOAT_WITHIN(1e-8, 1e-5f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-8, -1e-5f, out->data[1]);

    tensor_free(t);
    tensor_free(out);
}

void test_gelu_standard_values(void) {
    float32_t data[] = {0.0f, 1.0f, -1.0f, 2.0f};
    Tensor *t = create_tensor_from_data(data, 4);
    Tensor *out = tensor_gelu(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.845795f, out->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, -0.154205f, out->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 1.9356f, out->data[3]);

    tensor_free(t);
    tensor_free(out);
}

void test_gelu_stability(void) {
    float32_t data[] = {100.0f, -100.0f};
    Tensor *t = create_tensor_from_data(data, 2);
    Tensor *out = tensor_gelu(t);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 100.0f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, out->data[1]);

    tensor_free(t);
    tensor_free(out);
}

void test_gelu_nan_inf(void) {
    float32_t data[] = {NAN, -NAN};
    Tensor *t = create_tensor_from_data(data, 2);
    Tensor *out = tensor_gelu(t);

    TEST_ASSERT_TRUE(isnan(out->data[0]));
    TEST_ASSERT_TRUE(isnan(out->data[1]));

    tensor_free(t);
    tensor_free(out);
}

void test_gelu_inf(void) {
    float32_t data[] = {INFINITY, -INFINITY};
    Tensor *t = create_tensor_from_data(data, 2);
    Tensor *out = tensor_gelu(t);

    TEST_ASSERT_TRUE(isinf(out->data[0]) && out->data[0] > 0);
    TEST_ASSERT_TRUE(isnan(out->data[1]));

    tensor_free(t);
    tensor_free(out);
}

void test_softmax_standard(void) {
    float32_t data[] = {1.0f, 2.0f, 3.0f};
    uint64_t shape[] = {3};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *out = tensor_softmax(t, 0);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.09003057f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.24472847f, out->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.66524096f, out->data[2]);

    float32_t sum = 0;
    for (int i = 0; i < 3; i++)
        sum += out->data[i];
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, sum);

    tensor_free(t);
    tensor_free(out);
}

void test_softmax_shift_invariance_large_values(void) {
    float32_t data[] = {1000.0f, 1001.0f, 1002.0f};
    uint64_t shape[] = {3};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *out = tensor_softmax(t, 0);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.09003057f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.24472847f, out->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.66524096f, out->data[2]);

    tensor_free(t);
    tensor_free(out);
}

void test_softmax_large_negative_values(void) {
    float32_t data[] = {-1000.0f, -2000.0f, -3000.0f};
    uint64_t shape[] = {3};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *out = tensor_softmax(t, 0);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, out->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, out->data[2]);

    tensor_free(t);
    tensor_free(out);
}

void test_softmax_zero_tensor(void) {
    uint64_t shape[] = {3};
    Tensor *t = tensor_zeros(shape, 1, false);
    Tensor *out = tensor_softmax(t, 0);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.3333333f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.3333333f, out->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.3333333f, out->data[2]);

    tensor_free(t);
    tensor_free(out);
}

void test_softmax_2d_axis_0(void) {
    float32_t data[] = {1.0f, 10.0f, 2.0f, 10.0f};
    uint64_t shape[] = {2, 2};
    Tensor *t = tensor_create(data, shape, 2, false);
    Tensor *out = tensor_softmax(t, 0);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.268941f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.5f, out->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.731059f, out->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.5f, out->data[3]);

    tensor_free(t);
    tensor_free(out);
}

void test_softmax_nan(void) {
    float32_t data[] = {1.0f, NAN, 2.0f};
    Tensor *t = create_tensor_from_data(data, 3);
    Tensor *out = tensor_softmax(t, 0);

    TEST_ASSERT_TRUE(isnan(out->data[0]));
    TEST_ASSERT_TRUE(isnan(out->data[1]));
    TEST_ASSERT_TRUE(isnan(out->data[2]));

    tensor_free(t);
    tensor_free(out);
}

void test_softmax_3d_tensor(void) {
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    uint64_t shape[] = {2, 2, 2};
    Tensor *t = tensor_create(data, shape, 3, false);
    Tensor *out = tensor_softmax(t, 2);

    for (int i = 0; i < 4; i++) {
        float32_t v1 = out->data[2 * i];
        float32_t v2 = out->data[2 * i + 1];
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.2689414f, v1);
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.7310586f, v2);
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, v1 + v2);
    }

    tensor_free(t);
    tensor_free(out);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_sigmoid_standard_values);
    RUN_TEST(test_sigmoid_stability_large_positive);
    RUN_TEST(test_sigmoid_stability_large_negative);
    RUN_TEST(test_sigmoid_nan_inf);
    RUN_TEST(test_sigmoid_tiny_values);
    RUN_TEST(test_relu_standard_values);
    RUN_TEST(test_relu_stability);
    RUN_TEST(test_relu_nan_inf);
    RUN_TEST(test_relu_mixed_sign);
    RUN_TEST(test_tanh_standard_values);
    RUN_TEST(test_tanh_stability);
    RUN_TEST(test_tanh_nan_inf);
    RUN_TEST(test_tanh_small_epsilon);
    RUN_TEST(test_gelu_standard_values);
    RUN_TEST(test_gelu_stability);
    RUN_TEST(test_gelu_nan_inf);
    RUN_TEST(test_gelu_inf);
    RUN_TEST(test_softmax_standard);
    RUN_TEST(test_softmax_shift_invariance_large_values);
    RUN_TEST(test_softmax_large_negative_values);
    RUN_TEST(test_softmax_zero_tensor);
    RUN_TEST(test_softmax_2d_axis_0);
    RUN_TEST(test_softmax_nan);
    RUN_TEST(test_softmax_3d_tensor);
    return UNITY_END();
}
