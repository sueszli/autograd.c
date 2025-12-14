#include "../src/activations.h"
#include "../src/tensor.h"
#include "unity.h"
#include <math.h>
#include <stdlib.h>

#define TOLERANCE 1e-4f

void setUp(void) {}
void tearDown(void) {}

void test_sigmoid(void) {
    float32_t data[] = {-2.0f, 0.0f, 2.0f};
    uint64_t shape[] = {3};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *res = tensor_sigmoid(t);

    TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, 0.5f, res->data[1]);

    TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, 0.1192029f, res->data[0]);

    TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, 0.880797f, res->data[2]);

    tensor_free(t);
    tensor_free(res);

    float32_t extreme_data[] = {-1000.0f, 1000.0f};
    uint64_t extreme_shape[] = {2};
    Tensor *extreme_t = tensor_create(extreme_data, extreme_shape, 1, false);
    Tensor *extreme_res = tensor_sigmoid(extreme_t);

    TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, 0.0f, extreme_res->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, 1.0f, extreme_res->data[1]);

    tensor_free(extreme_t);
    tensor_free(extreme_res);
}

void test_relu(void) {
    float32_t data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    uint64_t shape[] = {5};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *res = tensor_relu(t);

    float32_t expected[] = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};
    for (int i = 0; i < 5; i++) {
        TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, expected[i], res->data[i]);
    }

    tensor_free(t);
    tensor_free(res);
}

void test_tanh(void) {
    float32_t data[] = {-2.0f, 0.0f, 2.0f};
    uint64_t shape[] = {3};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *res = tensor_tanh(t);

    TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, 0.0f, res->data[1]);

    TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, 0.96402f, res->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, -0.96402f, res->data[0]);

    tensor_free(t);
    tensor_free(res);
}

void test_gelu(void) {
    float32_t data[] = {-1.0f, 0.0f, 1.0f};
    uint64_t shape[] = {3};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *res = tensor_gelu(t);

    TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, 0.0f, res->data[1]);

    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.8413f, res->data[2]);

    TEST_ASSERT_FLOAT_WITHIN(0.001f, -0.1587f, res->data[0]);

    tensor_free(t);
    tensor_free(res);
}

void test_softmax(void) {
    float32_t data[] = {1.0f, 2.0f, 3.0f};
    uint64_t shape[] = {3};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *res = tensor_softmax(t, 0);

    float32_t sum = 0.0f;
    for (int i = 0; i < 3; i++) sum += res->data[i];
    TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, 1.0f, sum);

    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0900f, res->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.2447f, res->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.6652f, res->data[2]);

    tensor_free(t);
    tensor_free(res);

    float32_t data2d[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint64_t shape2d[] = {2, 2};
    Tensor *t2 = tensor_create(data2d, shape2d, 2, false);
    Tensor *res2 = tensor_softmax(t2, 1);

    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.2689f, res2->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.7310f, res2->data[1]);

    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.2689f, res2->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.7310f, res2->data[3]);

    tensor_free(t2);
    tensor_free(res2);
}

void test_relu_nan(void) {
    float32_t data[] = {NAN, 1.0f, -1.0f};
    uint64_t shape[] = {3};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *res = tensor_relu(t);

    TEST_ASSERT_TRUE(isnan(res->data[0]));
    TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, 1.0f, res->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, 0.0f, res->data[2]);

    tensor_free(t);
    tensor_free(res);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_sigmoid);
    RUN_TEST(test_relu);
    RUN_TEST(test_tanh);
    RUN_TEST(test_gelu);
    RUN_TEST(test_softmax);
    RUN_TEST(test_relu_nan);
    return UNITY_END();
}
