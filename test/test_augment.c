#include "../src/tensor.h"
#include "../src/utils/augment.h"
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

    random_horizontal_flip_mut(t, 0.0f);

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

    random_horizontal_flip_mut(t, 1.0f);

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

    random_horizontal_flip_mut(t, 1.0f);

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

    random_horizontal_flip_mut(t3, 1.0f);

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

    random_crop_mut(t, 2, 2, 1);

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

    random_crop_mut(t_chw, 2, 2, 1);

    TEST_ASSERT_EQUAL_UINT64(3, t_chw->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, t_chw->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, t_chw->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(2, t_chw->shape[2]);
    tensor_free(t_chw);
}

static void test_random_crop_3d_hwc(void) {
    uint64_t shape_hwc[] = {5, 4, 5};
    Tensor *t_hwc = tensor_zeros(shape_hwc, 3, false);

    random_crop_mut(t_hwc, 2, 2, 1);

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

    random_crop_mut(t, 2, 2, 0);

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

    random_crop_mut(t, 2, 2, 0);

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

    random_crop_mut(t, 3, 3, 1);

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

    random_horizontal_flip_mut(t, 1.0f);
    random_crop_mut(t, 2, 2, 0);

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

static void test_flip_p1_1x1(void) {
    uint64_t shape[] = {1, 1};
    Tensor *t = tensor_zeros(shape, 2, false);
    t->data[0] = 42.0f;
    random_horizontal_flip_mut(t, 1.0f);
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 42.0f, t->data[0]);
    tensor_free(t);
}

static void test_flip_p1_1x5(void) {
    uint64_t shape[] = {1, 5};
    Tensor *t = tensor_zeros(shape, 2, false);
    for (int i = 0; i < 5; i++)
        t->data[i] = (float)i;
    random_horizontal_flip_mut(t, 1.0f);
    float expected[] = {4, 3, 2, 1, 0};
    for (int i = 0; i < 5; i++) {
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected[i], t->data[i]);
    }
    tensor_free(t);
}

static void test_flip_p1_5x1(void) {
    uint64_t shape[] = {5, 1};
    Tensor *t = tensor_zeros(shape, 2, false);
    for (int i = 0; i < 5; i++)
        t->data[i] = (float)i;
    random_horizontal_flip_mut(t, 1.0f);
    for (int i = 0; i < 5; i++) {
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, (float)i, t->data[i]);
    }
    tensor_free(t);
}

static void test_flip_p1_reversibility(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_zeros(shape, 2, false);
    for (int i = 0; i < 6; i++)
        t->data[i] = (float)i;

    random_horizontal_flip_mut(t, 1.0f);
    random_horizontal_flip_mut(t, 1.0f);

    for (int i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, (float)i, t->data[i]);
    }
    tensor_free(t);
}

static void test_flip_p0_no_op(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_zeros(shape, 2, false);
    for (int i = 0; i < 6; i++)
        t->data[i] = (float)i;

    random_horizontal_flip_mut(t, 0.0f);

    for (int i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, (float)i, t->data[i]);
    }
    tensor_free(t);
}

static void test_flip_p0_pointer_check(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_zeros(shape, 2, false);
    float32_t *original_ptr = t->data;

    random_horizontal_flip_mut(t, 0.0f);

    TEST_ASSERT_EQUAL_PTR(original_ptr, t->data);
    tensor_free(t);
}

static void test_flip_p1_pointer_check(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_zeros(shape, 2, false);

    random_horizontal_flip_mut(t, 1.0f);

    TEST_ASSERT_NOT_NULL(t->data);
    TEST_ASSERT_EQUAL_UINT64(2, t->ndim);
    tensor_free(t);
}

static void test_flip_invariants_sum(void) {
    uint64_t shape[] = {3, 3};
    Tensor *t = tensor_zeros(shape, 2, false);
    float sum_orig = 0;
    for (int i = 0; i < 9; i++) {
        t->data[i] = (float)i;
        sum_orig += (float)i;
    }

    random_horizontal_flip_mut(t, 1.0f);

    float sum_new = 0;
    for (int i = 0; i < 9; i++)
        sum_new += t->data[i];

    TEST_ASSERT_FLOAT_WITHIN(EPSILON, sum_orig, sum_new);
    tensor_free(t);
}

static void test_flip_4d_tensor(void) {
    uint64_t shape[] = {2, 2, 2, 3};
    Tensor *t = tensor_zeros(shape, 4, false);
    int cnt = 0;
    for (int i = 0; i < 24; i++)
        t->data[i] = (float)cnt++;

    random_horizontal_flip_mut(t, 1.0f);

    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f, t->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, t->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, t->data[2]);
    tensor_free(t);
}

static void test_flip_large_tensor(void) {
    uint64_t shape[] = {100, 100};
    Tensor *t = tensor_zeros(shape, 2, false);
    for (int i = 0; i < 10000; i++)
        t->data[i] = (float)i;

    random_horizontal_flip_mut(t, 1.0f);

    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 99.0f, t->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, t->data[99]);
    tensor_free(t);
}

static void test_crop_2d_to_1x1(void) {
    uint64_t shape[] = {4, 4};
    Tensor *t = tensor_zeros(shape, 2, false);
    random_crop_mut(t, 1, 1, 0);
    TEST_ASSERT_EQUAL_UINT64(1, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(1, t->shape[1]);
    tensor_free(t);
}

static void test_crop_2d_same_size(void) {
    uint64_t shape[] = {4, 4};
    Tensor *t = tensor_zeros(shape, 2, false);
    t->data[0] = 42.0f;
    random_crop_mut(t, 4, 4, 0);
    TEST_ASSERT_EQUAL_UINT64(4, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(4, t->shape[1]);
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 42.0f, t->data[0]);
    tensor_free(t);
}

static void test_crop_chw_boundary_c4(void) {
    uint64_t shape[] = {4, 5, 5};
    Tensor *t = tensor_zeros(shape, 3, false);
    random_crop_mut(t, 3, 3, 0);
    TEST_ASSERT_EQUAL_UINT64(4, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, t->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(3, t->shape[2]);
    tensor_free(t);
}

static void test_crop_hwc_boundary_h5(void) {
    uint64_t shape[] = {5, 5, 3};
    Tensor *t = tensor_zeros(shape, 3, false);
    random_crop_mut(t, 3, 3, 0);
    TEST_ASSERT_EQUAL_UINT64(3, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, t->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(3, t->shape[2]);
    tensor_free(t);
}

static void test_crop_padding_expansion(void) {
    uint64_t shape[] = {2, 2};
    Tensor *t = tensor_zeros(shape, 2, false);
    for (int i = 0; i < 4; i++)
        t->data[i] = 1.0f;

    random_crop_mut(t, 4, 4, 2);

    TEST_ASSERT_EQUAL_UINT64(4, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(4, t->shape[1]);
    float sum = 0;
    for (int i = 0; i < 16; i++)
        sum += t->data[i];
    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 4.0f, sum);
    tensor_free(t);
}

static void test_crop_extreme_padding(void) {
    uint64_t shape[] = {2, 2};
    Tensor *t = tensor_zeros(shape, 2, false);
    t->data[0] = 1.0f;

    random_crop_mut(t, 2, 2, 100);

    TEST_ASSERT_EQUAL_UINT64(2, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[1]);

    tensor_free(t);
}

static void test_crop_2d_strides(void) {
    uint64_t shape[] = {4, 4};
    Tensor *t = tensor_zeros(shape, 2, false);
    random_crop_mut(t, 2, 2, 0);
    TEST_ASSERT_EQUAL_UINT64(2, t->strides[0]);
    TEST_ASSERT_EQUAL_UINT64(1, t->strides[1]);
    tensor_free(t);
}

static void test_crop_3d_chw_strides(void) {
    uint64_t shape[] = {3, 4, 4};
    Tensor *t = tensor_zeros(shape, 3, false);
    random_crop_mut(t, 2, 2, 0);

    TEST_ASSERT_EQUAL_UINT64(4, t->strides[0]);
    TEST_ASSERT_EQUAL_UINT64(2, t->strides[1]);
    TEST_ASSERT_EQUAL_UINT64(1, t->strides[2]);
    tensor_free(t);
}

static void test_crop_3d_hwc_strides(void) {
    uint64_t shape[] = {5, 5, 3};
    Tensor *t = tensor_zeros(shape, 3, false);
    random_crop_mut(t, 2, 2, 0);

    TEST_ASSERT_EQUAL_UINT64(6, t->strides[0]);
    TEST_ASSERT_EQUAL_UINT64(3, t->strides[1]);
    TEST_ASSERT_EQUAL_UINT64(1, t->strides[2]);
    tensor_free(t);
}

static void test_crop_valid_data_range(void) {
    uint64_t shape[] = {4, 4};
    Tensor *t = tensor_zeros(shape, 2, false);
    for (int i = 0; i < 16; i++)
        t->data[i] = 5.0f;

    random_crop_mut(t, 2, 2, 1);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_TRUE(t->data[i] == 5.0f || t->data[i] == 0.0f);
    }
    tensor_free(t);
}

static void test_crop_constant_input_no_pad(void) {
    uint64_t shape[] = {4, 4};
    Tensor *t = tensor_zeros(shape, 2, false);
    for (int i = 0; i < 16; i++)
        t->data[i] = 7.0f;

    random_crop_mut(t, 2, 2, 0);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, 7.0f, t->data[i]);
    }
    tensor_free(t);
}

static void test_crop_constant_input_with_pad(void) {
    uint64_t shape[] = {2, 2};
    Tensor *t = tensor_zeros(shape, 2, false);
    for (int i = 0; i < 4; i++)
        t->data[i] = 1.0f;

    random_crop_mut(t, 2, 2, 1);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_TRUE(t->data[i] == 1.0f || t->data[i] == 0.0f);
    }
    tensor_free(t);
}

static void test_crop_all_zeros(void) {
    uint64_t shape[] = {2, 2};
    Tensor *t = tensor_zeros(shape, 2, false);
    random_crop_mut(t, 2, 2, 1);
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, 0.0f, t->data[i]);
    }
    tensor_free(t);
}

static void test_chain_flip_crop(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_zeros(shape, 2, false);
    for (int i = 0; i < 6; i++)
        t->data[i] = (float)i;

    random_horizontal_flip_mut(t, 1.0f);

    random_crop_mut(t, 2, 2, 0);

    TEST_ASSERT_EQUAL_UINT64(2, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[1]);
    tensor_free(t);
}

static void test_chain_crop_flip(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_zeros(shape, 2, false);
    for (int i = 0; i < 6; i++)
        t->data[i] = (float)i;

    random_crop_mut(t, 2, 2, 0);
    random_horizontal_flip_mut(t, 1.0f);

    TEST_ASSERT_EQUAL_UINT64(2, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[1]);
    tensor_free(t);
}

static void test_crop_shape_preserved_chw(void) {
    uint64_t shape[] = {3, 4, 4};
    Tensor *t = tensor_zeros(shape, 3, false);
    random_crop_mut(t, 2, 2, 0);
    TEST_ASSERT_EQUAL_UINT64(3, t->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[2]);
    tensor_free(t);
}

static void test_crop_shape_preserved_hwc(void) {
    uint64_t shape[] = {5, 5, 3};
    Tensor *t = tensor_zeros(shape, 3, false);
    random_crop_mut(t, 2, 2, 0);
    TEST_ASSERT_EQUAL_UINT64(3, t->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(3, t->shape[2]);
    tensor_free(t);
}

static void test_flip_affects_data(void) {
    uint64_t shape[] = {1, 3};
    Tensor *t = tensor_zeros(shape, 2, false);
    t->data[0] = 0;
    t->data[1] = 1;
    t->data[2] = 2;
    random_horizontal_flip_mut(t, 1.0f);

    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 2.0f, t->data[0]);
    tensor_free(t);
}

static void test_crop_affects_data(void) {
    uint64_t shape[] = {2, 2};
    Tensor *t = tensor_zeros(shape, 2, false);
    t->data[0] = 1;
    t->data[1] = 2;
    t->data[2] = 3;
    t->data[3] = 4;

    random_crop_mut(t, 1, 1, 0);
    TEST_ASSERT_EQUAL_UINT64(1, t->size);

    bool found = (t->data[0] == 1 || t->data[0] == 2 || t->data[0] == 3 || t->data[0] == 4);
    TEST_ASSERT_TRUE(found);
    tensor_free(t);
}

static void test_augment_idempotency_p0(void) {
    uint64_t shape[] = {2, 2};
    Tensor *t = tensor_zeros(shape, 2, false);
    t->data[0] = 1;
    t->data[1] = 2;
    t->data[2] = 3;
    t->data[3] = 4;

    random_horizontal_flip_mut(t, 0.0f);
    random_horizontal_flip_mut(t, 0.0f);

    TEST_ASSERT_FLOAT_WITHIN(EPSILON, 1.0f, t->data[0]);
    tensor_free(t);
}

static void test_crop_partial_overlap(void) {
    uint64_t shape[] = {2, 2};
    Tensor *t = tensor_zeros(shape, 2, false);
    for (int i = 0; i < 4; i++)
        t->data[i] = 1.0f;

    random_crop_mut(t, 2, 2, 1);
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[1]);
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
    RUN_TEST(test_flip_p1_1x1);
    RUN_TEST(test_flip_p1_1x5);
    RUN_TEST(test_flip_p1_5x1);
    RUN_TEST(test_flip_p1_reversibility);
    RUN_TEST(test_flip_p0_no_op);
    RUN_TEST(test_flip_p0_pointer_check);
    RUN_TEST(test_flip_p1_pointer_check);
    RUN_TEST(test_flip_invariants_sum);
    RUN_TEST(test_flip_4d_tensor);
    RUN_TEST(test_flip_large_tensor);
    RUN_TEST(test_crop_2d_to_1x1);
    RUN_TEST(test_crop_2d_same_size);
    RUN_TEST(test_crop_chw_boundary_c4);
    RUN_TEST(test_crop_hwc_boundary_h5);
    RUN_TEST(test_crop_padding_expansion);
    RUN_TEST(test_crop_extreme_padding);
    RUN_TEST(test_crop_2d_strides);
    RUN_TEST(test_crop_3d_chw_strides);
    RUN_TEST(test_crop_3d_hwc_strides);
    RUN_TEST(test_crop_valid_data_range);
    RUN_TEST(test_crop_constant_input_no_pad);
    RUN_TEST(test_crop_constant_input_with_pad);
    RUN_TEST(test_crop_all_zeros);
    RUN_TEST(test_chain_flip_crop);
    RUN_TEST(test_chain_crop_flip);
    RUN_TEST(test_crop_shape_preserved_chw);
    RUN_TEST(test_crop_shape_preserved_hwc);
    RUN_TEST(test_flip_affects_data);
    RUN_TEST(test_crop_affects_data);
    RUN_TEST(test_augment_idempotency_p0);
    RUN_TEST(test_crop_partial_overlap);
    return UNITY_END();
}
