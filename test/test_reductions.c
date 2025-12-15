#include "ops/reductions.h"
#include "tensor.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_tensor_sum(void) {
    uint64_t shape1[] = {3};
    float32_t data1[] = {1.0f, 2.0f, 3.0f};
    Tensor *t1 = tensor_create(data1, shape1, 1, false);

    Tensor *sum1 = tensor_sum(t1, 0, false);
    TEST_ASSERT_EQUAL_UINT64(0, sum1->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, sum1->size);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 6.0f, sum1->data[0]);
    tensor_free(sum1);

    Tensor *sum2 = tensor_sum(t1, 0, true);
    TEST_ASSERT_EQUAL_UINT64(1, sum2->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, sum2->shape[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 6.0f, sum2->data[0]);
    tensor_free(sum2);

    tensor_free(t1);

    uint64_t shape2[] = {2, 3};
    float32_t data2[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor *t2 = tensor_create(data2, shape2, 2, false);

    Tensor *sum_dim0 = tensor_sum(t2, 0, false);
    TEST_ASSERT_EQUAL_UINT64(1, sum_dim0->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, sum_dim0->shape[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 5.0f, sum_dim0->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 7.0f, sum_dim0->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 9.0f, sum_dim0->data[2]);
    tensor_free(sum_dim0);

    Tensor *sum_dim1 = tensor_sum(t2, 1, false);
    TEST_ASSERT_EQUAL_UINT64(1, sum_dim1->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, sum_dim1->shape[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 6.0f, sum_dim1->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 15.0f, sum_dim1->data[1]);
    tensor_free(sum_dim1);

    tensor_free(t2);
}

void test_tensor_mean(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor *t = tensor_create(data, shape, 2, false);

    Tensor *mean_dim0 = tensor_mean(t, 0, false);
    TEST_ASSERT_EQUAL_UINT64(1, mean_dim0->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, mean_dim0->shape[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 2.5f, mean_dim0->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.5f, mean_dim0->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 4.5f, mean_dim0->data[2]);
    tensor_free(mean_dim0);

    Tensor *mean_dim1 = tensor_mean(t, 1, false);
    TEST_ASSERT_EQUAL_UINT64(1, mean_dim1->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, mean_dim1->shape[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 2.0f, mean_dim1->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 5.0f, mean_dim1->data[1]);
    tensor_free(mean_dim1);

    tensor_free(t);
}

void test_tensor_max(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f};
    Tensor *t = tensor_create(data, shape, 2, false);

    Tensor *max_dim0 = tensor_max(t, 0, false);
    TEST_ASSERT_EQUAL_UINT64(1, max_dim0->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, max_dim0->shape[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 4.0f, max_dim0->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 5.0f, max_dim0->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 6.0f, max_dim0->data[2]);
    tensor_free(max_dim0);

    Tensor *max_dim1 = tensor_max(t, 1, false);
    TEST_ASSERT_EQUAL_UINT64(1, max_dim1->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, max_dim1->shape[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 5.0f, max_dim1->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 6.0f, max_dim1->data[1]);
    tensor_free(max_dim1);

    tensor_free(t);
}

void test_sum_3d_dim0(void) {
    uint64_t shape[] = {2, 2, 2};
    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor *t = tensor_create(data, shape, 3, false);
    Tensor *res = tensor_sum(t, 0, false);
    TEST_ASSERT_EQUAL_UINT64(2, res->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, res->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, res->shape[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 6.0f, res->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 8.0f, res->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 10.0f, res->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 12.0f, res->data[3]);
    tensor_free(res);
    tensor_free(t);
}

void test_sum_3d_dim1(void) {
    uint64_t shape[] = {2, 2, 2};
    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor *t = tensor_create(data, shape, 3, false);
    Tensor *res = tensor_sum(t, 1, false);
    TEST_ASSERT_EQUAL_UINT64(2, res->ndim);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 4.0f, res->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 6.0f, res->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 12.0f, res->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 14.0f, res->data[3]);
    tensor_free(res);
    tensor_free(t);
}

void test_sum_3d_dim2(void) {
    uint64_t shape[] = {2, 2, 2};
    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor *t = tensor_create(data, shape, 3, false);
    Tensor *res = tensor_sum(t, 2, false);
    TEST_ASSERT_EQUAL_UINT64(2, res->ndim);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.0f, res->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 7.0f, res->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 11.0f, res->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 15.0f, res->data[3]);
    tensor_free(res);
    tensor_free(t);
}

void test_sum_3d_dim0_keepdims(void) {
    uint64_t shape[] = {2, 2, 2};
    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor *t = tensor_create(data, shape, 3, false);
    Tensor *res = tensor_sum(t, 0, true);
    TEST_ASSERT_EQUAL_UINT64(3, res->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, res->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, res->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(2, res->shape[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 6.0f, res->data[0]);
    tensor_free(res);
    tensor_free(t);
}

void test_mean_3d_dim0(void) {
    uint64_t shape[] = {2, 2, 2};
    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor *t = tensor_create(data, shape, 3, false);
    Tensor *res = tensor_mean(t, 0, false);
    TEST_ASSERT_EQUAL_UINT64(2, res->ndim);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.0f, res->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 4.0f, res->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 5.0f, res->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 6.0f, res->data[3]);
    tensor_free(res);
    tensor_free(t);
}

void test_mean_3d_dim1(void) {
    uint64_t shape[] = {2, 2, 2};
    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor *t = tensor_create(data, shape, 3, false);
    Tensor *res = tensor_mean(t, 1, false);
    TEST_ASSERT_EQUAL_UINT64(2, res->ndim);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 2.0f, res->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.0f, res->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 6.0f, res->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 7.0f, res->data[3]);
    tensor_free(res);
    tensor_free(t);
}

void test_mean_3d_dim2(void) {
    uint64_t shape[] = {2, 2, 2};
    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor *t = tensor_create(data, shape, 3, false);
    Tensor *res = tensor_mean(t, 2, false);
    TEST_ASSERT_EQUAL_UINT64(2, res->ndim);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.5f, res->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.5f, res->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 5.5f, res->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 7.5f, res->data[3]);
    tensor_free(res);
    tensor_free(t);
}

void test_mean_3d_dim0_keepdims(void) {
    uint64_t shape[] = {2, 2, 2};
    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor *t = tensor_create(data, shape, 3, false);
    Tensor *res = tensor_mean(t, 0, true);
    TEST_ASSERT_EQUAL_UINT64(3, res->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, res->shape[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.0f, res->data[0]);
    tensor_free(res);
    tensor_free(t);
}

void test_max_3d_dim0(void) {
    uint64_t shape[] = {2, 2, 2};
    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor *t = tensor_create(data, shape, 3, false);
    Tensor *res = tensor_max(t, 0, false);
    TEST_ASSERT_EQUAL_UINT64(2, res->ndim);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 5.0f, res->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 6.0f, res->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 7.0f, res->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 8.0f, res->data[3]);
    tensor_free(res);
    tensor_free(t);
}

void test_max_3d_dim1(void) {
    uint64_t shape[] = {2, 2, 2};
    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor *t = tensor_create(data, shape, 3, false);
    Tensor *res = tensor_max(t, 1, false);
    TEST_ASSERT_EQUAL_UINT64(2, res->ndim);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.0f, res->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 4.0f, res->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 7.0f, res->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 8.0f, res->data[3]);
    tensor_free(res);
    tensor_free(t);
}

void test_max_3d_dim2(void) {
    uint64_t shape[] = {2, 2, 2};
    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor *t = tensor_create(data, shape, 3, false);
    Tensor *res = tensor_max(t, 2, false);
    TEST_ASSERT_EQUAL_UINT64(2, res->ndim);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 2.0f, res->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 4.0f, res->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 6.0f, res->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 8.0f, res->data[3]);
    tensor_free(res);
    tensor_free(t);
}

void test_max_3d_dim0_keepdims(void) {
    uint64_t shape[] = {2, 2, 2};
    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor *t = tensor_create(data, shape, 3, false);
    Tensor *res = tensor_max(t, 0, true);
    TEST_ASSERT_EQUAL_UINT64(3, res->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, res->shape[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 5.0f, res->data[0]);
    tensor_free(res);
    tensor_free(t);
}

void test_sum_scalar(void) {
    uint64_t shape[] = {1};
    float32_t data[] = {42.0f};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *res = tensor_sum(t, 0, false);
    TEST_ASSERT_EQUAL_UINT64(0, res->ndim);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 42.0f, res->data[0]);
    tensor_free(res);
    tensor_free(t);
}

void test_mean_scalar(void) {
    uint64_t shape[] = {1};
    float32_t data[] = {42.0f};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *res = tensor_mean(t, 0, false);
    TEST_ASSERT_EQUAL_UINT64(0, res->ndim);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 42.0f, res->data[0]);
    tensor_free(res);
    tensor_free(t);
}

void test_max_scalar(void) {
    uint64_t shape[] = {1};
    float32_t data[] = {42.0f};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *res = tensor_max(t, 0, false);
    TEST_ASSERT_EQUAL_UINT64(0, res->ndim);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 42.0f, res->data[0]);
    tensor_free(res);
    tensor_free(t);
}

void test_sum_large_dim(void) {
    uint64_t shape[] = {100};
    Tensor *t = tensor_zeros(shape, 1, false);
    for (int i = 0; i < 100; i++)
        t->data[i] = 1.0f;
    Tensor *res = tensor_sum(t, 0, false);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 100.0f, res->data[0]);
    tensor_free(res);
    tensor_free(t);
}

void test_mean_large_dim(void) {
    uint64_t shape[] = {100};
    Tensor *t = tensor_zeros(shape, 1, false);
    for (int i = 0; i < 100; i++)
        t->data[i] = 2.0f;
    Tensor *res = tensor_mean(t, 0, false);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 2.0f, res->data[0]);
    tensor_free(res);
    tensor_free(t);
}

void test_sum_negative(void) {
    uint64_t shape[] = {4};
    float32_t data[] = {10.0f, -5.0f, 2.0f, -8.0f};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *res = tensor_sum(t, 0, false);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -1.0f, res->data[0]);
    tensor_free(res);
    tensor_free(t);
}

void test_max_negative(void) {
    uint64_t shape[] = {4};
    float32_t data[] = {-10.0f, -5.0f, -2.0f, -8.0f};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *res = tensor_max(t, 0, false);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -2.0f, res->data[0]);
    tensor_free(res);
    tensor_free(t);
}

void test_mean_negative(void) {
    uint64_t shape[] = {4};
    float32_t data[] = {-10.0f, -5.0f, -2.0f, -7.0f};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *res = tensor_mean(t, 0, false);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -6.0f, res->data[0]);
    tensor_free(res);
    tensor_free(t);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_tensor_sum);
    RUN_TEST(test_tensor_mean);
    RUN_TEST(test_tensor_max);
    RUN_TEST(test_sum_3d_dim0);
    RUN_TEST(test_sum_3d_dim1);
    RUN_TEST(test_sum_3d_dim2);
    RUN_TEST(test_sum_3d_dim0_keepdims);
    RUN_TEST(test_mean_3d_dim0);
    RUN_TEST(test_mean_3d_dim1);
    RUN_TEST(test_mean_3d_dim2);
    RUN_TEST(test_mean_3d_dim0_keepdims);
    RUN_TEST(test_max_3d_dim0);
    RUN_TEST(test_max_3d_dim1);
    RUN_TEST(test_max_3d_dim2);
    RUN_TEST(test_max_3d_dim0_keepdims);
    RUN_TEST(test_sum_scalar);
    RUN_TEST(test_mean_scalar);
    RUN_TEST(test_max_scalar);
    RUN_TEST(test_sum_large_dim);
    RUN_TEST(test_mean_large_dim);
    RUN_TEST(test_sum_negative);
    RUN_TEST(test_max_negative);
    RUN_TEST(test_mean_negative);
    return UNITY_END();
}
