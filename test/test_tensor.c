#include "../src/tensor.h"
#include "unity.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

void test_tensor_create(void) {
    int64_t shape[] = {2, 3};
    Tensor *t = tensor_zeros(shape, 2, false);
    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_INT64(2, t->ndim);
    TEST_ASSERT_EQUAL_INT64(2, t->shape[0]);
    TEST_ASSERT_EQUAL_INT64(3, t->shape[1]);
    TEST_ASSERT_EQUAL_INT64(6, t->size);
    TEST_ASSERT_EQUAL_INT64(3, t->strides[0]);
    TEST_ASSERT_EQUAL_INT64(1, t->strides[1]);

    for (int64_t i = 0; i < 6; i++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, t->data[i]);
    }

    tensor_free(t);
}

void test_tensor_create_scalar(void) {
    float32_t data = 5.0f;
    Tensor *t = tensor_create(&data, NULL, 0, false);
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
    Tensor *a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {5, 6, 7, 8};
    int64_t shape_b[] = {2, 2};
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    Tensor *c = tensor_add(a, b);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL_INT64(2, c->ndim);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, c->data[0]);  // 1+5
    TEST_ASSERT_EQUAL_FLOAT(8.0f, c->data[1]);  // 2+6
    TEST_ASSERT_EQUAL_FLOAT(10.0f, c->data[2]); // 3+7
    TEST_ASSERT_EQUAL_FLOAT(12.0f, c->data[3]); // 4+8

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_broadcast_add(void) {
    float32_t data_a[] = {1, 2};
    int64_t shape_a[] = {2, 1};
    Tensor *a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {10, 20};
    int64_t shape_b[] = {2};
    Tensor *b = tensor_create(data_b, shape_b, 1, false);

    // a: (2, 1) [[1], [2]]
    // b: (2)    [10, 20] -> broadcasts to (1, 2) [[10, 20]]
    // result: (2, 2) [[11, 21], [12, 22]]

    Tensor *c = tensor_add(a, b);
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
    Tensor *a = tensor_create(data_a, shape_a, 1, false);

    float32_t val = 10.0f;
    Tensor *s = tensor_create(&val, NULL, 0, false);

    Tensor *c = tensor_add(a, s);
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
    Tensor *a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {2, 5, 2, 5};
    int64_t shape_b[] = {2, 2};
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    Tensor *sub = tensor_sub(a, b);
    TEST_ASSERT_EQUAL_FLOAT(8.0f, sub->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(15.0f, sub->data[1]);
    tensor_free(sub);

    Tensor *mul = tensor_mul(a, b);
    TEST_ASSERT_EQUAL_FLOAT(20.0f, mul->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(100.0f, mul->data[1]);
    tensor_free(mul);

    Tensor *div = tensor_div(a, b);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, div->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, div->data[1]);
    tensor_free(div);

    tensor_free(a);
    tensor_free(b);
}

void test_tensor_matmul(void) {
    float32_t data_a[] = {1, 2, 3, 4, 5, 6}; // 2x3
    int64_t shape_a[] = {2, 3};
    Tensor *a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {7, 8, 9, 1, 2, 3}; // 3x2
    int64_t shape_b[] = {3, 2};
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    Tensor *c = tensor_matmul(a, b);
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
    Tensor *t = tensor_create(data, shape, 2, false);

    int64_t new_shape[] = {3, 2};
    Tensor *t2 = tensor_reshape(t, new_shape, 2);

    TEST_ASSERT_NOT_NULL(t2);
    TEST_ASSERT_EQUAL_INT64(3, t2->shape[0]);
    TEST_ASSERT_EQUAL_INT64(2, t2->shape[1]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, t2->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, t2->data[5]);

    int64_t new_shape_auto[] = {6, -1};
    Tensor *t3 = tensor_reshape(t, new_shape_auto, 2);
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
    Tensor *t = tensor_create(data, shape, 2, false);

    Tensor *t2 = tensor_transpose(t, 0, 1);
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
    Tensor *t = tensor_create(data, shape, 2, false);

    // sum axis 0: [5, 7, 9] (shape 3)
    Tensor *s0 = tensor_sum(t, 0, false);
    TEST_ASSERT_NOT_NULL(s0);
    TEST_ASSERT_EQUAL_INT64(1, s0->ndim);
    TEST_ASSERT_EQUAL_INT64(3, s0->shape[0]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, s0->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(7.0f, s0->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(9.0f, s0->data[2]);
    tensor_free(s0);

    // sum axis 1: [6, 15] (shape 2)
    Tensor *s1 = tensor_sum(t, 1, false);
    TEST_ASSERT_NOT_NULL(s1);
    TEST_ASSERT_EQUAL_INT64(1, s1->ndim);
    TEST_ASSERT_EQUAL_INT64(2, s1->shape[0]);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, s1->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(15.0f, s1->data[1]);
    tensor_free(s1);

    // mean axis 1: [2, 5]
    Tensor *m1 = tensor_mean(t, 1, false);
    TEST_ASSERT_NOT_NULL(m1);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, m1->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, m1->data[1]);
    tensor_free(m1);

    // max axis 1: [3, 6]
    Tensor *max1 = tensor_max(t, 1, false);
    TEST_ASSERT_NOT_NULL(max1);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, max1->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, max1->data[1]);
    tensor_free(max1);

    // test keepdims
    Tensor *s1_kd = tensor_sum(t, 1, true);
    TEST_ASSERT_EQUAL_INT64(2, s1_kd->ndim);
    TEST_ASSERT_EQUAL_INT64(2, s1_kd->shape[0]);
    TEST_ASSERT_EQUAL_INT64(1, s1_kd->shape[1]);
    tensor_free(s1_kd);

    tensor_free(t);
}

void test_broadcast_error(void) {
    float32_t data_a[] = {1, 2};
    int64_t shape_a[] = {2};
    Tensor *a = tensor_create(data_a, shape_a, 1, false);

    float32_t data_b[] = {1, 2, 3};
    int64_t shape_b[] = {3};
    Tensor *b = tensor_create(data_b, shape_b, 1, false);

    Tensor *c = tensor_add(a, b);
    TEST_ASSERT_NULL(c);

    tensor_free(a);
    tensor_free(b);
}

void test_matmul_error_shapes(void) {
    float32_t data_a[] = {1, 2, 3, 4}; // 2x2
    int64_t shape_a[] = {2, 2};
    Tensor *a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {1, 2, 3, 4}; // 2x2
    int64_t shape_b[] = {2, 2};
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    // mismatched k dim
    int64_t shape_bad[] = {3, 2};
    Tensor *c = tensor_reshape(b, shape_bad, 2); // this will fail logic actually because size 4 != 6
    TEST_ASSERT_NULL(c);

    // correct fail test: inner dims don't match
    // A: 2x2
    // B: 3x1 (size 3)
    float32_t data_d[] = {1, 2, 3};
    int64_t shape_d[] = {3, 1};
    Tensor *d = tensor_create(data_d, shape_d, 2, false);

    Tensor *res = tensor_matmul(a, d);
    TEST_ASSERT_NULL(res);

    tensor_free(a);
    tensor_free(b);
    tensor_free(d);
}

void test_tensor_get(void) {
    // 2x3 tensor
    float32_t data[] = {1, 2, 3, 4, 5, 6};
    int64_t shape[] = {2, 3};
    Tensor *t = tensor_create(data, shape, 2, false);

    // get element at [0, 1] -> 2.0
    int64_t idx1[] = {0, 1};
    Tensor *val1 = tensor_get(t, idx1);
    TEST_ASSERT_NOT_NULL(val1);
    TEST_ASSERT_EQUAL_INT64(0, val1->ndim);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, val1->data[0]);
    tensor_free(val1);

    // get element at [1, 2] -> 6.0
    int64_t idx2[] = {1, 2};
    Tensor *val2 = tensor_get(t, idx2);
    TEST_ASSERT_NOT_NULL(val2);
    TEST_ASSERT_EQUAL_INT64(0, val2->ndim);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, val2->data[0]);
    tensor_free(val2);

    tensor_free(t);
}

void test_tensor_requires_grad(void) {
    int64_t shape[] = {2};
    Tensor *t = tensor_zeros(shape, 1, true);
    TEST_ASSERT_TRUE(t->requires_grad);
    TEST_ASSERT_NULL(t->grad); // Should be NULL initially

    Tensor *t_no_grad = tensor_zeros(shape, 1, false);
    TEST_ASSERT_FALSE(t_no_grad->requires_grad);

    tensor_free(t);
    tensor_free(t_no_grad);
}

void test_tensor_broadcast_complex(void) {
    // A: (3, 1) -> [[1], [2], [3]]
    float32_t data_a[] = {1, 2, 3};
    int64_t shape_a[] = {3, 1};
    Tensor *a = tensor_create(data_a, shape_a, 2, false);

    // B: (1, 4) -> [[10, 20, 30, 40]]
    float32_t data_b[] = {10, 20, 30, 40};
    int64_t shape_b[] = {1, 4};
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    // expected result: (3, 4)
    // [[11, 21, 31, 41],
    //  [12, 22, 32, 42],
    //  [13, 23, 33, 43]]

    Tensor *c = tensor_add(a, b);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL_INT64(2, c->ndim);
    TEST_ASSERT_EQUAL_INT64(3, c->shape[0]);
    TEST_ASSERT_EQUAL_INT64(4, c->shape[1]);

    TEST_ASSERT_EQUAL_FLOAT(11.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(13.0f, c->data[8]);  // row 2, col 0 (index 8) -> 3 + 10 = 13
    TEST_ASSERT_EQUAL_FLOAT(43.0f, c->data[11]); // row 2, col 3 (index 11) -> 3 + 40 = 43

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_reshape_errors(void) {
    float32_t data[] = {1, 2, 3, 4};
    int64_t shape[] = {4};
    Tensor *t = tensor_create(data, shape, 1, false);

    // error: multiple -1
    int64_t shape_err1[] = {-1, -1};
    Tensor *t_err1 = tensor_reshape(t, shape_err1, 2);
    TEST_ASSERT_NULL(t_err1);

    // error: total size mismatch
    int64_t shape_err2[] = {2, 3}; // size 6 != 4
    Tensor *t_err2 = tensor_reshape(t, shape_err2, 2);
    TEST_ASSERT_NULL(t_err2);

    // error: -1 but not divisible
    int64_t shape_err3[] = {3, -1}; // 4 is not divisible by 3
    Tensor *t_err3 = tensor_reshape(t, shape_err3, 2);
    TEST_ASSERT_NULL(t_err3);

    tensor_free(t);
}

void test_tensor_transpose_general(void) {
    // 3D tensor: (2, 3, 2)
    // data: 0..11
    float32_t data[12];
    for (int i = 0; i < 12; i++)
        data[i] = (float32_t)i;
    int64_t shape[] = {2, 3, 2};
    Tensor *t = tensor_create(data, shape, 3, false);

    // transpose axis 0 and 2 -> shape (2, 3, 2)
    // original strides: [6, 2, 1]
    // T[i, j, k] -> T[k, j, i]
    Tensor *t2 = tensor_transpose(t, 0, 2);
    TEST_ASSERT_NOT_NULL(t2);
    TEST_ASSERT_EQUAL_INT64(2, t2->shape[0]);
    TEST_ASSERT_EQUAL_INT64(3, t2->shape[1]);
    TEST_ASSERT_EQUAL_INT64(2, t2->shape[2]);

    // check value at [0, 1, 1] in new tensor
    // corresponds to [1, 1, 0] in old tensor
    // old: 1*6 + 1*2 + 0*1 = 8 -> data[8] = 8.0
    // new index: 0*6 + 1*2 + 1*1 = 3
    TEST_ASSERT_EQUAL_FLOAT(8.0f, t2->data[3]);

    tensor_free(t);
    tensor_free(t2);
}

void test_tensor_div_broadcast(void) {
    float32_t data_a[] = {10, 20, 30, 40};
    int64_t shape_a[] = {2, 2};
    Tensor *a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {2};
    int64_t shape_b[] = {1};
    Tensor *b = tensor_create(data_b, shape_b, 1, false);

    Tensor *c = tensor_div(a, b);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(20.0f, c->data[3]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
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
    RUN_TEST(test_broadcast_error);
    RUN_TEST(test_matmul_error_shapes);
    RUN_TEST(test_tensor_get);
    RUN_TEST(test_tensor_requires_grad);
    RUN_TEST(test_tensor_broadcast_complex);
    RUN_TEST(test_tensor_reshape_errors);
    RUN_TEST(test_tensor_transpose_general);
    RUN_TEST(test_tensor_div_broadcast);
    return UNITY_END();
}
