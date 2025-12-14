#include "unity.h"
#include "../src/tensor.h"
#include <stdlib.h>
#include <math.h>

void setUp(void) {}
void tearDown(void) {}

void test_tensor_create(void) {
    int64_t shape[] = {2, 3};
    Tensor* t = tensor_zeros(shape, 2, false);
    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_INT64(2, t->ndim);
    TEST_ASSERT_EQUAL_INT64(2, t->shape[0]);
    TEST_ASSERT_EQUAL_INT64(3, t->shape[1]);
    TEST_ASSERT_EQUAL_INT64(6, t->size);
    TEST_ASSERT_EQUAL_INT64(3, t->strides[0]);
    TEST_ASSERT_EQUAL_INT64(1, t->strides[1]);

    for(int64_t i=0; i<6; i++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, t->data[i]);
    }

    tensor_free(t);
}

void test_tensor_create_scalar(void) {
    float32_t data = 5.0f;
    Tensor* t = tensor_create(&data, NULL, 0, false);
    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_INT64(0, t->ndim);
    TEST_ASSERT_NULL(t->shape);
    TEST_ASSERT_EQUAL_INT64(1, t->size);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, t->data[0]);
    tensor_free(t);
}

void test_tensor_add(void) {
    float32_t data_a[] = {1, 2, 3, 4};
    int64_t shape_a[] = {2, 2};
    Tensor* a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {5, 6, 7, 8};
    int64_t shape_b[] = {2, 2};
    Tensor* b = tensor_create(data_b, shape_b, 2, false);

    Tensor* c = tensor_add(a, b);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL_INT64(2, c->ndim);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(8.0f, c->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(10.0f, c->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(12.0f, c->data[3]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_broadcast_add(void) {
    float32_t data_a[] = {1, 2};
    int64_t shape_a[] = {2, 1};
    Tensor* a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {10, 20};
    int64_t shape_b[] = {2};
    Tensor* b = tensor_create(data_b, shape_b, 1, false);

    Tensor* c = tensor_add(a, b);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL_INT64(2, c->ndim);
    TEST_ASSERT_EQUAL_INT64(2, c->shape[0]);
    TEST_ASSERT_EQUAL_INT64(2, c->shape[1]);

    TEST_ASSERT_EQUAL_FLOAT(11.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(21.0f, c->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(12.0f, c->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(22.0f, c->data[3]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_scalar_ops(void) {
    float32_t data_a[] = {1, 2, 3};
    int64_t shape_a[] = {3};
    Tensor* a = tensor_create(data_a, shape_a, 1, false);

    float32_t val = 10.0f;
    Tensor* s = tensor_create(&val, NULL, 0, false);

    Tensor* c = tensor_add(a, s);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL_INT64(1, c->ndim);
    TEST_ASSERT_EQUAL_FLOAT(11.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(12.0f, c->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(13.0f, c->data[2]);

    tensor_free(a);
    tensor_free(s);
    tensor_free(c);
}

void test_tensor_sub_mul_div(void) {
    float32_t data_a[] = {10, 20, 30, 40};
    int64_t shape_a[] = {2, 2};
    Tensor* a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {2, 5, 2, 5};
    int64_t shape_b[] = {2, 2};
    Tensor* b = tensor_create(data_b, shape_b, 2, false);

    Tensor* sub = tensor_sub(a, b);
    TEST_ASSERT_EQUAL_FLOAT(8.0f, sub->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(15.0f, sub->data[1]);
    tensor_free(sub);

    Tensor* mul = tensor_mul(a, b);
    TEST_ASSERT_EQUAL_FLOAT(20.0f, mul->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(100.0f, mul->data[1]);
    tensor_free(mul);

    Tensor* div = tensor_div(a, b);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, div->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, div->data[1]);
    tensor_free(div);

    tensor_free(a);
    tensor_free(b);
}

void test_tensor_matmul(void) {
    float32_t data_a[] = {1, 2, 3, 4, 5, 6}; // 2x3
    int64_t shape_a[] = {2, 3};
    Tensor* a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {7, 8, 9, 1, 2, 3}; // 3x2
    int64_t shape_b[] = {3, 2};
    Tensor* b = tensor_create(data_b, shape_b, 2, false);

    Tensor* c = tensor_matmul(a, b);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL_INT64(2, c->ndim);
    TEST_ASSERT_EQUAL_INT64(2, c->shape[0]);
    TEST_ASSERT_EQUAL_INT64(2, c->shape[1]);

    TEST_ASSERT_EQUAL_FLOAT(31.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(19.0f, c->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(85.0f, c->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(55.0f, c->data[3]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_reshape(void) {
    float32_t data[] = {1, 2, 3, 4, 5, 6};
    int64_t shape[] = {2, 3};
    Tensor* t = tensor_create(data, shape, 2, false);

    int64_t new_shape[] = {3, 2};
    Tensor* t2 = tensor_reshape(t, new_shape, 2);

    TEST_ASSERT_NOT_NULL(t2);
    TEST_ASSERT_EQUAL_INT64(3, t2->shape[0]);
    TEST_ASSERT_EQUAL_INT64(2, t2->shape[1]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, t2->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, t2->data[5]);

    int64_t new_shape_auto[] = {6, -1};
    Tensor* t3 = tensor_reshape(t, new_shape_auto, 2);
    TEST_ASSERT_NOT_NULL(t3);
    TEST_ASSERT_EQUAL_INT64(6, t3->shape[0]);
    TEST_ASSERT_EQUAL_INT64(1, t3->shape[1]);

    tensor_free(t);
    tensor_free(t2);
    tensor_free(t3);
}

void test_tensor_transpose(void) {
    float32_t data[] = {1, 2, 3, 4, 5, 6}; // 2x3
    int64_t shape[] = {2, 3};
    Tensor* t = tensor_create(data, shape, 2, false);

    Tensor* t2 = tensor_transpose(t, 0, 1);
    TEST_ASSERT_NOT_NULL(t2);
    TEST_ASSERT_EQUAL_INT64(3, t2->shape[0]);
    TEST_ASSERT_EQUAL_INT64(2, t2->shape[1]);

    TEST_ASSERT_EQUAL_FLOAT(1.0f, t2->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, t2->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, t2->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, t2->data[3]);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, t2->data[4]);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, t2->data[5]);

    tensor_free(t);
    tensor_free(t2);
}

void test_tensor_reductions(void) {
    float32_t data[] = {1, 2, 3, 4, 5, 6}; // 2x3
    int64_t shape[] = {2, 3};
    Tensor* t = tensor_create(data, shape, 2, false);

    // sum axis 0
    Tensor* s0 = tensor_sum(t, 0, false);
    TEST_ASSERT_NOT_NULL(s0);
    TEST_ASSERT_EQUAL_INT64(1, s0->ndim);
    TEST_ASSERT_EQUAL_INT64(3, s0->shape[0]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, s0->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(7.0f, s0->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(9.0f, s0->data[2]);
    tensor_free(s0);

    // sum axis 1
    Tensor* s1 = tensor_sum(t, 1, false);
    TEST_ASSERT_NOT_NULL(s1);
    TEST_ASSERT_EQUAL_INT64(1, s1->ndim);
    TEST_ASSERT_EQUAL_INT64(2, s1->shape[0]);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, s1->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(15.0f, s1->data[1]);
    tensor_free(s1);

    // mean axis 1
    Tensor* m1 = tensor_mean(t, 1, false);
    TEST_ASSERT_NOT_NULL(m1);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, m1->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, m1->data[1]);
    tensor_free(m1);

    // max axis 1
    Tensor* max1 = tensor_max(t, 1, false);
    TEST_ASSERT_NOT_NULL(max1);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, max1->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, max1->data[1]);
    tensor_free(max1);

    // test keepdims
    Tensor* s1_kd = tensor_sum(t, 1, true);
    TEST_ASSERT_EQUAL_INT64(2, s1_kd->ndim);
    TEST_ASSERT_EQUAL_INT64(2, s1_kd->shape[0]);
    TEST_ASSERT_EQUAL_INT64(1, s1_kd->shape[1]);
    tensor_free(s1_kd);

    tensor_free(t);
}

void test_tensor_get(void) {
    float32_t data[] = {1, 2, 3, 4};
    int64_t shape[] = {2, 2};
    Tensor* t = tensor_create(data, shape, 2, false);

    int64_t indices[] = {1, 0};
    Tensor* val = tensor_get(t, indices);
    TEST_ASSERT_NOT_NULL(val);
    TEST_ASSERT_EQUAL_INT64(0, val->ndim);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, val->data[0]);

    tensor_free(val);
    tensor_free(t);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_tensor_create);
    RUN_TEST(test_tensor_create_scalar);
    RUN_TEST(test_tensor_add);
    RUN_TEST(test_tensor_broadcast_add);
    RUN_TEST(test_tensor_scalar_ops);
    RUN_TEST(test_tensor_sub_mul_div);
    RUN_TEST(test_tensor_matmul);
    RUN_TEST(test_tensor_reshape);
    RUN_TEST(test_tensor_transpose);
    RUN_TEST(test_tensor_reductions);
    RUN_TEST(test_tensor_get);
    return UNITY_END();
}
