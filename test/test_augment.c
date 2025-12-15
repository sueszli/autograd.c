#include "../src/augment.h"
#include "../src/tensor.h"
#include "unity.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void setUp(void) { srand(42); }
void tearDown(void) {}

#define EPSILON 1e-5

static void test_random_horizontal_flip_p0(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_zeros(shape, 2, false);
    float32_t data[] = {0, 1, 2, 3, 4, 5};
    memcpy(t->data, data, 6 * sizeof(float32_t));

    random_horizontal_flip(t, 0.0f);

    for (int i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, data[i], t->data[i]);
    }
    tensor_free(t);
}

static void test_random_horizontal_flip_p1(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_zeros(shape, 2, false);
    float32_t data[] = {0, 1, 2, 3, 4, 5};
    memcpy(t->data, data, 6 * sizeof(float32_t));

    random_horizontal_flip(t, 1.0f);

    float32_t expected[] = {2, 1, 0, 5, 4, 3};
    for (int i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected[i], t->data[i]);
    }
    tensor_free(t);
}

static void test_random_horizontal_flip_odd_width(void) {
    uint64_t shape[] = {1, 3};
    Tensor *t = tensor_zeros(shape, 2, false);
    float32_t data[] = {0, 1, 2};
    memcpy(t->data, data, 3 * sizeof(float32_t));

    random_horizontal_flip(t, 1.0f);

    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f, t->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, t->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, t->data[2]);
    tensor_free(t);
}

static void test_random_horizontal_flip_3d(void) {
    uint64_t shape3[] = {2, 2, 3};
    Tensor *t3 = tensor_zeros(shape3, 3, false);
    for (int i = 0; i < 12; i++)
        t3->data[i] = (float32_t)i;

    random_horizontal_flip(t3, 1.0f);

    float32_t expected3[] = {2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9};
    for (int i = 0; i < 12; i++) {
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected3[i], t3->data[i]);
    }
    tensor_free(t3);
}

static void test_random_crop_2d(void) {
    uint64_t shape[] = {4, 4};
    Tensor *t = tensor_zeros(shape, 2, false);
    for (int i = 0; i < 16; i++)
        t->data[i] = 1.0f;

    random_crop(t, 2, 2, 1);

    TEST_ASSERT_EQUAL_UINT64(2, t->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[1]);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_TRUE(t->data[i] == 0.0f || t->data[i] == 1.0f);
    }
    tensor_free(t);
}

static void test_random_crop_3d_chw(void) {
    uint64_t shape_chw[] = {3, 4, 4};
    Tensor *t_chw = tensor_zeros(shape_chw, 3, false);
    for (int i = 0; i < 3 * 4 * 4; i++)
        t_chw->data[i] = 1.0f;

    random_crop(t_chw, 2, 2, 1);

    TEST_ASSERT_EQUAL_UINT64(3, t_chw->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, t_chw->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, t_chw->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(2, t_chw->shape[2]);
    tensor_free(t_chw);
}

static void test_random_crop_3d_hwc(void) {
    uint64_t shape_hwc[] = {5, 4, 5};
    Tensor *t_hwc = tensor_zeros(shape_hwc, 3, false);

    random_crop(t_hwc, 2, 2, 1);

    TEST_ASSERT_EQUAL_UINT64(3, t_hwc->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, t_hwc->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, t_hwc->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(5, t_hwc->shape[2]);
    tensor_free(t_hwc);
}

static void test_random_crop_identity(void) {
    uint64_t shape[] = {2, 2};
    Tensor *t = tensor_zeros(shape, 2, false);
    t->data[0] = 1.0f;
    t->data[1] = 2.0f;
    t->data[2] = 3.0f;
    t->data[3] = 4.0f;

    random_crop(t, 2, 2, 0);

    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, t->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f, t->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, t->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, t->data[3]);
    tensor_free(t);
}

static void test_random_crop_zero_padding(void) {
    uint64_t shape[] = {3, 3};
    Tensor *t = tensor_zeros(shape, 2, false);
    for (int i = 0; i < 9; i++)
        t->data[i] = 1.0f;

    random_crop(t, 2, 2, 0);

    TEST_ASSERT_EQUAL_UINT64(2, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[1]);
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, t->data[i]);
    }
    tensor_free(t);
}

static void test_random_crop_larger_than_input(void) {
    uint64_t shape[] = {1, 1};
    Tensor *t = tensor_zeros(shape, 2, false);
    t->data[0] = 9.0f;

    random_crop(t, 3, 3, 1);

    TEST_ASSERT_EQUAL_UINT64(3, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, t->shape[1]);

    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 9.0f, t->data[4]);
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, t->data[0]);
    tensor_free(t);
}

static void test_chaining(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_zeros(shape, 2, false);
    float32_t data[] = {0, 1, 2, 3, 4, 5};
    memcpy(t->data, data, 6 * sizeof(float32_t));

    random_horizontal_flip(t, 1.0f);
    random_crop(t, 2, 2, 0);

    TEST_ASSERT_EQUAL_UINT64(2, t->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[1]);

    float32_t v0 = t->data[0];
    float32_t v1 = t->data[1];
    float32_t v2 = t->data[2];
    float32_t v3 = t->data[3];

    if (fabs(v0 - 2.0f) < EPSILON) {
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, v1);
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, 5.0f, v2);
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, v3);
    } else {
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, v0);
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, v1);
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, v2);
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, 3.0f, v3);
    }

    tensor_free(t);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_random_horizontal_flip_p0);
    RUN_TEST(test_random_horizontal_flip_p1);
    RUN_TEST(test_random_horizontal_flip_odd_width);
    RUN_TEST(test_random_horizontal_flip_3d);
    RUN_TEST(test_random_crop_2d);
    RUN_TEST(test_random_crop_3d_chw);
    RUN_TEST(test_random_crop_3d_hwc);
    RUN_TEST(test_random_crop_identity);
    RUN_TEST(test_random_crop_zero_padding);
    RUN_TEST(test_random_crop_larger_than_input);
    RUN_TEST(test_chaining);
    return UNITY_END();
}
