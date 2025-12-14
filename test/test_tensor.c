#include "unity.h"
#include "../src/tensor.h"
#include <stdlib.h>
#include <math.h>

void setUp(void) {}
void tearDown(void) {}

void test_tensor_create(void) {
    int shape[] = {2, 3};
    Tensor* t = tensor_zeros(shape, 2, false);
    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL(2, t->ndim);
    TEST_ASSERT_EQUAL(2, t->shape[0]);
    TEST_ASSERT_EQUAL(3, t->shape[1]);
    TEST_ASSERT_EQUAL(6, t->size);
    TEST_ASSERT_EQUAL(3, t->strides[0]);
    TEST_ASSERT_EQUAL(1, t->strides[1]);

    for(int i=0; i<6; i++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, t->data[i]);
    }

    tensor_free(t);
}

void test_tensor_create_scalar(void) {
    float data = 5.0f;
    Tensor* t = tensor_create(&data, NULL, 0, false);
    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL(0, t->ndim);
    TEST_ASSERT_NULL(t->shape);
    TEST_ASSERT_EQUAL(1, t->size);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, t->data[0]);
    tensor_free(t);
}

void test_tensor_add(void) {
    float data_a[] = {1, 2, 3, 4};
    int shape_a[] = {2, 2};
    Tensor* a = tensor_create(data_a, shape_a, 2, false);

    float data_b[] = {5, 6, 7, 8};
    int shape_b[] = {2, 2};
    Tensor* b = tensor_create(data_b, shape_b, 2, false);

    Tensor* c = tensor_add(a, b);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL(2, c->ndim);
    TEST_ASSERT_EQUAL(6.0f, c->data[0]); // 1+5
    TEST_ASSERT_EQUAL(8.0f, c->data[1]); // 2+6
    TEST_ASSERT_EQUAL(10.0f, c->data[2]); // 3+7
    TEST_ASSERT_EQUAL(12.0f, c->data[3]); // 4+8

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_broadcast_add(void) {
    float data_a[] = {1, 2};
    int shape_a[] = {2, 1};
    Tensor* a = tensor_create(data_a, shape_a, 2, false);

    float data_b[] = {10, 20};
    int shape_b[] = {2};
    Tensor* b = tensor_create(data_b, shape_b, 1, false);

    // a: (2, 1) [[1], [2]]
    // b: (2)    [10, 20] -> broadcasts to (1, 2) [[10, 20]]
    // result: (2, 2) [[11, 21], [12, 22]]

    Tensor* c = tensor_add(a, b);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL(2, c->ndim);
    TEST_ASSERT_EQUAL(2, c->shape[0]);
    TEST_ASSERT_EQUAL(2, c->shape[1]);

    TEST_ASSERT_EQUAL_FLOAT(11.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(21.0f, c->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(12.0f, c->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(22.0f, c->data[3]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_scalar_ops(void) {
    float data_a[] = {1, 2, 3};
    int shape_a[] = {3};
    Tensor* a = tensor_create(data_a, shape_a, 1, false);

    float val = 10.0f;
    Tensor* s = tensor_create(&val, NULL, 0, false);

    Tensor* c = tensor_add(a, s);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL(1, c->ndim);
    TEST_ASSERT_EQUAL(11.0f, c->data[0]);
    TEST_ASSERT_EQUAL(12.0f, c->data[1]);
    TEST_ASSERT_EQUAL(13.0f, c->data[2]);

    tensor_free(a);
    tensor_free(s);
    tensor_free(c);
}

void test_tensor_sub_mul_div(void) {
    float data_a[] = {10, 20, 30, 40};
    int shape_a[] = {2, 2};
    Tensor* a = tensor_create(data_a, shape_a, 2, false);

    float data_b[] = {2, 5, 2, 5};
    int shape_b[] = {2, 2};
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
    float data_a[] = {1, 2, 3, 4, 5, 6}; // 2x3
    int shape_a[] = {2, 3};
    Tensor* a = tensor_create(data_a, shape_a, 2, false);

    float data_b[] = {7, 8, 9, 1, 2, 3}; // 3x2
    int shape_b[] = {3, 2};
    Tensor* b = tensor_create(data_b, shape_b, 2, false);

    // a @ b
    // [[1, 2, 3], [4, 5, 6]] @ [[7, 8], [9, 1], [2, 3]]
    // [0,0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    // [0,1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    // [1,0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    // [1,1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55

    Tensor* c = tensor_matmul(a, b);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL(2, c->ndim);
    TEST_ASSERT_EQUAL(2, c->shape[0]);
    TEST_ASSERT_EQUAL(2, c->shape[1]);

    TEST_ASSERT_EQUAL_FLOAT(31.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(19.0f, c->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(85.0f, c->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(55.0f, c->data[3]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_reshape(void) {
    float data[] = {1, 2, 3, 4, 5, 6};
    int shape[] = {2, 3};
    Tensor* t = tensor_create(data, shape, 2, false);

    int new_shape[] = {3, 2};
    Tensor* t2 = tensor_reshape(t, new_shape, 2);

    TEST_ASSERT_NOT_NULL(t2);
    TEST_ASSERT_EQUAL(3, t2->shape[0]);
    TEST_ASSERT_EQUAL(2, t2->shape[1]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, t2->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, t2->data[5]);

    int new_shape_auto[] = {6, -1};
    Tensor* t3 = tensor_reshape(t, new_shape_auto, 2);
    TEST_ASSERT_NOT_NULL(t3);
    TEST_ASSERT_EQUAL(6, t3->shape[0]);
    TEST_ASSERT_EQUAL(1, t3->shape[1]);

    tensor_free(t);
    tensor_free(t2);
    tensor_free(t3);
}

void test_tensor_transpose(void) {
    float data[] = {1, 2, 3, 4, 5, 6}; // 2x3
    int shape[] = {2, 3};
    Tensor* t = tensor_create(data, shape, 2, false);

    Tensor* t2 = tensor_transpose(t, 0, 1);
    TEST_ASSERT_NOT_NULL(t2);
    TEST_ASSERT_EQUAL(3, t2->shape[0]);
    TEST_ASSERT_EQUAL(2, t2->shape[1]);

    // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
    // data: 1, 4, 2, 5, 3, 6
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
    float data[] = {1, 2, 3, 4, 5, 6}; // 2x3
    int shape[] = {2, 3};
    Tensor* t = tensor_create(data, shape, 2, false);

    // Sum axis 0: [5, 7, 9] (shape 3)
    Tensor* s0 = tensor_sum(t, 0, false);
    TEST_ASSERT_NOT_NULL(s0);
    TEST_ASSERT_EQUAL(1, s0->ndim);
    TEST_ASSERT_EQUAL(3, s0->shape[0]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, s0->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(7.0f, s0->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(9.0f, s0->data[2]);
    tensor_free(s0);

    // Sum axis 1: [6, 15] (shape 2)
    Tensor* s1 = tensor_sum(t, 1, false);
    TEST_ASSERT_NOT_NULL(s1);
    TEST_ASSERT_EQUAL(1, s1->ndim);
    TEST_ASSERT_EQUAL(2, s1->shape[0]);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, s1->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(15.0f, s1->data[1]);
    tensor_free(s1);

    // Mean axis 1: [2, 5]
    Tensor* m1 = tensor_mean(t, 1, false);
    TEST_ASSERT_NOT_NULL(m1);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, m1->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, m1->data[1]);
    tensor_free(m1);

    // Max axis 1: [3, 6]
    Tensor* max1 = tensor_max(t, 1, false);
    TEST_ASSERT_NOT_NULL(max1);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, max1->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, max1->data[1]);
    tensor_free(max1);

    // Test keepdims
    Tensor* s1_kd = tensor_sum(t, 1, true);
    TEST_ASSERT_EQUAL(2, s1_kd->ndim);
    TEST_ASSERT_EQUAL(2, s1_kd->shape[0]);
    TEST_ASSERT_EQUAL(1, s1_kd->shape[1]);
    tensor_free(s1_kd);

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
    return UNITY_END();
}
