#include "tensor.h"
#include "unity.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

void test_tensor_create(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_zeros(shape, 2, false);
    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(2, t->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, t->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(6, t->size);
    TEST_ASSERT_EQUAL_UINT64(3, t->strides[0]);
    TEST_ASSERT_EQUAL_UINT64(1, t->strides[1]);

    for (uint64_t i = 0; i < 6; i++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, t->data[i]);
    }

    tensor_free(t);
}

void test_tensor_create_scalar(void) {
    float32_t data = 5.0f;
    Tensor *t = tensor_create(&data, NULL, 0, false);
    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(0, t->ndim);
    TEST_ASSERT_NULL(t->shape);
    TEST_ASSERT_EQUAL_UINT64(1, t->size);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, t->data[0]);
    tensor_free(t);
}

void test_tensor_add(void) {
    float32_t data_a[] = {1, 2, 3, 4};
    uint64_t shape_a[] = {2, 2};
    Tensor *a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {5, 6, 7, 8};
    uint64_t shape_b[] = {2, 2};
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    Tensor *c = tensor_add(a, b);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL_UINT64(2, c->ndim);
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
    uint64_t shape_a[] = {2, 1};
    Tensor *a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {10, 20};
    uint64_t shape_b[] = {2};
    Tensor *b = tensor_create(data_b, shape_b, 1, false);

    Tensor *c = tensor_add(a, b);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL_UINT64(2, c->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, c->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, c->shape[1]);

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
    uint64_t shape_a[] = {3};
    Tensor *a = tensor_create(data_a, shape_a, 1, false);

    float32_t val = 10.0f;
    Tensor *s = tensor_create(&val, NULL, 0, false);

    Tensor *c = tensor_add(a, s);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL_UINT64(1, c->ndim);
    TEST_ASSERT_EQUAL_FLOAT(11.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(12.0f, c->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(13.0f, c->data[2]);

    tensor_free(a);
    tensor_free(s);
    tensor_free(c);
}

void test_tensor_sub_mul_div(void) {
    float32_t data_a[] = {10, 20, 30, 40};
    uint64_t shape_a[] = {2, 2};
    Tensor *a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {2, 5, 2, 5};
    uint64_t shape_b[] = {2, 2};
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
    float32_t data_a[] = {1, 2, 3, 4, 5, 6};
    uint64_t shape_a[] = {2, 3};
    Tensor *a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {7, 8, 9, 1, 2, 3};
    uint64_t shape_b[] = {3, 2};
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    Tensor *c = tensor_matmul(a, b);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL_UINT64(2, c->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, c->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, c->shape[1]);

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
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_create(data, shape, 2, false);

    int64_t new_shape[] = {3, 2};
    Tensor *t2 = tensor_reshape(t, new_shape, 2);

    TEST_ASSERT_NOT_NULL(t2);
    TEST_ASSERT_EQUAL_UINT64(3, t2->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, t2->shape[1]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, t2->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, t2->data[5]);

    int64_t new_shape_auto[] = {6, -1};
    Tensor *t3 = tensor_reshape(t, new_shape_auto, 2);
    TEST_ASSERT_NOT_NULL(t3);
    TEST_ASSERT_EQUAL_UINT64(6, t3->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(1, t3->shape[1]);

    tensor_free(t);
    tensor_free(t2);
    tensor_free(t3);
}

void test_tensor_transpose(void) {
    float32_t data[] = {1, 2, 3, 4, 5, 6};
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_create(data, shape, 2, false);

    Tensor *t2 = tensor_transpose(t, 0, 1);
    TEST_ASSERT_NOT_NULL(t2);
    TEST_ASSERT_EQUAL_UINT64(3, t2->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, t2->shape[1]);

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
    float32_t data[] = {1, 2, 3, 4, 5, 6};
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_create(data, shape, 2, false);

    Tensor *s0 = tensor_sum(t, 0, false);
    TEST_ASSERT_NOT_NULL(s0);
    TEST_ASSERT_EQUAL_UINT64(1, s0->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, s0->shape[0]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, s0->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(7.0f, s0->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(9.0f, s0->data[2]);
    tensor_free(s0);

    Tensor *s1 = tensor_sum(t, 1, false);
    TEST_ASSERT_NOT_NULL(s1);
    TEST_ASSERT_EQUAL_UINT64(1, s1->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, s1->shape[0]);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, s1->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(15.0f, s1->data[1]);
    tensor_free(s1);

    Tensor *m1 = tensor_mean(t, 1, false);
    TEST_ASSERT_NOT_NULL(m1);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, m1->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, m1->data[1]);
    tensor_free(m1);

    Tensor *max1 = tensor_max(t, 1, false);
    TEST_ASSERT_NOT_NULL(max1);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, max1->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, max1->data[1]);
    tensor_free(max1);

    Tensor *s1_kd = tensor_sum(t, 1, true);
    TEST_ASSERT_EQUAL_UINT64(2, s1_kd->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, s1_kd->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(1, s1_kd->shape[1]);
    tensor_free(s1_kd);

    tensor_free(t);
}

void test_tensor_get(void) {
    float32_t data[] = {1, 2, 3, 4, 5, 6};
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_create(data, shape, 2, false);

    uint64_t idx1[] = {0, 1};
    Tensor *val1 = tensor_get(t, idx1);
    TEST_ASSERT_NOT_NULL(val1);
    TEST_ASSERT_EQUAL_UINT64(0, val1->ndim);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, val1->data[0]);
    tensor_free(val1);

    uint64_t idx2[] = {1, 2};
    Tensor *val2 = tensor_get(t, idx2);
    TEST_ASSERT_NOT_NULL(val2);
    TEST_ASSERT_EQUAL_UINT64(0, val2->ndim);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, val2->data[0]);
    tensor_free(val2);

    tensor_free(t);
}

void test_tensor_requires_grad(void) {
    uint64_t shape[] = {2};
    Tensor *t = tensor_zeros(shape, 1, true);
    TEST_ASSERT_TRUE(t->requires_grad);
    TEST_ASSERT_NULL(t->grad);

    Tensor *t_no_grad = tensor_zeros(shape, 1, false);
    TEST_ASSERT_FALSE(t_no_grad->requires_grad);

    tensor_free(t);
    tensor_free(t_no_grad);
}

void test_tensor_broadcast_complex(void) {
    float32_t data_a[] = {1, 2, 3};
    uint64_t shape_a[] = {3, 1};
    Tensor *a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {10, 20, 30, 40};
    uint64_t shape_b[] = {1, 4};
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    Tensor *c = tensor_add(a, b);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL_UINT64(2, c->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, c->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(4, c->shape[1]);

    TEST_ASSERT_EQUAL_FLOAT(11.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(13.0f, c->data[8]);
    TEST_ASSERT_EQUAL_FLOAT(43.0f, c->data[11]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_transpose_general(void) {
    float32_t data[12];
    for (int i = 0; i < 12; i++)
        data[i] = (float32_t)i;
    uint64_t shape[] = {2, 3, 2};
    Tensor *t = tensor_create(data, shape, 3, false);

    Tensor *t2 = tensor_transpose(t, 0, 2);
    TEST_ASSERT_NOT_NULL(t2);
    TEST_ASSERT_EQUAL_UINT64(2, t2->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, t2->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(2, t2->shape[2]);

    TEST_ASSERT_EQUAL_FLOAT(8.0f, t2->data[3]);

    tensor_free(t);
    tensor_free(t2);
}

void test_tensor_div_broadcast(void) {
    float32_t data_a[] = {10, 20, 30, 40};
    uint64_t shape_a[] = {2, 2};
    Tensor *a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {2};
    uint64_t shape_b[] = {1};
    Tensor *b = tensor_create(data_b, shape_b, 1, false);

    Tensor *c = tensor_div(a, b);
    TEST_ASSERT_NOT_NULL(c);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(20.0f, c->data[3]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_create_1d(void) {
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    uint64_t shape[] = {5};
    Tensor *t = tensor_create(data, shape, 1, false);

    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(1, t->ndim);
    TEST_ASSERT_EQUAL_UINT64(5, t->size);
    TEST_ASSERT_EQUAL_UINT64(1, t->strides[0]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, t->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, t->data[4]);

    tensor_free(t);
}

void test_tensor_create_3d(void) {
    float32_t data[24];
    for (int i = 0; i < 24; i++) {
        data[i] = (float32_t)i;
    }
    uint64_t shape[] = {2, 3, 4};
    Tensor *t = tensor_create(data, shape, 3, false);

    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(3, t->ndim);
    TEST_ASSERT_EQUAL_UINT64(24, t->size);
    TEST_ASSERT_EQUAL_UINT64(12, t->strides[0]);
    TEST_ASSERT_EQUAL_UINT64(4, t->strides[1]);
    TEST_ASSERT_EQUAL_UINT64(1, t->strides[2]);

    tensor_free(t);
}

void test_tensor_create_4d(void) {
    uint64_t shape[] = {2, 2, 2, 2};
    Tensor *t = tensor_zeros(shape, 4, false);

    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(4, t->ndim);
    TEST_ASSERT_EQUAL_UINT64(16, t->size);
    TEST_ASSERT_EQUAL_UINT64(8, t->strides[0]);
    TEST_ASSERT_EQUAL_UINT64(4, t->strides[1]);
    TEST_ASSERT_EQUAL_UINT64(2, t->strides[2]);
    TEST_ASSERT_EQUAL_UINT64(1, t->strides[3]);

    tensor_free(t);
}

void test_tensor_create_single_element_non_scalar(void) {
    uint64_t shape[] = {1, 1, 1};
    float32_t data[] = {42.0f};
    Tensor *t = tensor_create(data, shape, 3, false);

    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(3, t->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, t->size);
    TEST_ASSERT_EQUAL_FLOAT(42.0f, t->data[0]);

    tensor_free(t);
}

void test_tensor_zeros_initializes_correctly(void) {
    uint64_t shape[] = {3, 4};
    Tensor *t = tensor_zeros(shape, 2, false);

    for (uint64_t i = 0; i < 12; i++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, t->data[i]);
    }

    tensor_free(t);
}

void test_tensor_strides_various_shapes(void) {
    uint64_t shape1[] = {5};
    Tensor *t1 = tensor_zeros(shape1, 1, false);
    TEST_ASSERT_EQUAL_UINT64(1, t1->strides[0]);
    tensor_free(t1);

    uint64_t shape2[] = {3, 5};
    Tensor *t2 = tensor_zeros(shape2, 2, false);
    TEST_ASSERT_EQUAL_UINT64(5, t2->strides[0]);
    TEST_ASSERT_EQUAL_UINT64(1, t2->strides[1]);
    tensor_free(t2);

    uint64_t shape3[] = {2, 3, 4, 5};
    Tensor *t3 = tensor_zeros(shape3, 4, false);
    TEST_ASSERT_EQUAL_UINT64(60, t3->strides[0]);
    TEST_ASSERT_EQUAL_UINT64(20, t3->strides[1]);
    TEST_ASSERT_EQUAL_UINT64(5, t3->strides[2]);
    TEST_ASSERT_EQUAL_UINT64(1, t3->strides[3]);
    tensor_free(t3);
}

void test_tensor_broadcast_scalar_to_tensor(void) {
    float32_t scalar_val = 5.0f;
    Tensor *scalar = tensor_create(&scalar_val, NULL, 0, false);

    float32_t data[] = {1, 2, 3, 4};
    uint64_t shape[] = {2, 2};
    Tensor *t = tensor_create(data, shape, 2, false);

    Tensor *result = tensor_mul(t, scalar);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, result->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(10.0f, result->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(15.0f, result->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(20.0f, result->data[3]);

    tensor_free(scalar);
    tensor_free(t);
    tensor_free(result);
}

void test_tensor_broadcast_different_ndims(void) {
    uint64_t shape_a[] = {2, 3, 4};
    Tensor *a = tensor_zeros(shape_a, 3, false);
    for (uint64_t i = 0; i < a->size; i++) {
        a->data[i] = 1.0f;
    }

    float32_t data_b[] = {1, 2, 3, 4};
    uint64_t shape_b[] = {4};
    Tensor *b = tensor_create(data_b, shape_b, 1, false);

    Tensor *result = tensor_mul(a, b);
    TEST_ASSERT_EQUAL_UINT64(3, result->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, result->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, result->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(4, result->shape[2]);

    TEST_ASSERT_EQUAL_FLOAT(1.0f, result->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, result->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, result->data[3]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
}

void test_tensor_broadcast_multiple_1_dims(void) {
    uint64_t shape_a[] = {3, 1, 5, 1};
    Tensor *a = tensor_zeros(shape_a, 4, false);
    for (uint64_t i = 0; i < a->size; i++) {
        a->data[i] = 2.0f;
    }

    uint64_t shape_b[] = {1, 4, 1, 6};
    Tensor *b = tensor_zeros(shape_b, 4, false);
    for (uint64_t i = 0; i < b->size; i++) {
        b->data[i] = 3.0f;
    }

    Tensor *result = tensor_add(a, b);
    TEST_ASSERT_EQUAL_UINT64(4, result->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, result->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(4, result->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(5, result->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(6, result->shape[3]);
    TEST_ASSERT_EQUAL_UINT64(360, result->size);

    for (uint64_t i = 0; i < result->size; i++) {
        TEST_ASSERT_EQUAL_FLOAT(5.0f, result->data[i]);
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
}

void test_tensor_matmul_square(void) {
    float32_t data_a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    uint64_t shape[] = {3, 3};
    Tensor *a = tensor_create(data_a, shape, 2, false);

    float32_t data_b[] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    Tensor *b = tensor_create(data_b, shape, 2, false);

    Tensor *c = tensor_matmul(a, b);
    TEST_ASSERT_EQUAL_FLOAT(30.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(24.0f, c->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(18.0f, c->data[2]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_matmul_1x1(void) {
    float32_t data_a[] = {5.0f};
    uint64_t shape_a[] = {1, 1};
    Tensor *a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {3.0f};
    uint64_t shape_b[] = {1, 1};
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    Tensor *c = tensor_matmul(a, b);
    TEST_ASSERT_EQUAL_UINT64(1, c->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(1, c->shape[1]);
    TEST_ASSERT_EQUAL_FLOAT(15.0f, c->data[0]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_matmul_identity(void) {
    float32_t data_identity[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    uint64_t shape[] = {3, 3};
    Tensor *identity = tensor_create(data_identity, shape, 2, false);

    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Tensor *a = tensor_create(data, shape, 2, false);

    Tensor *result = tensor_matmul(a, identity);

    for (int i = 0; i < 9; i++) {
        TEST_ASSERT_EQUAL_FLOAT(data[i], result->data[i]);
    }

    tensor_free(identity);
    tensor_free(a);
    tensor_free(result);
}

void test_tensor_matmul_tall_and_wide(void) {
    float32_t data_a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint64_t shape_a[] = {4, 2};
    Tensor *a = tensor_create(data_a, shape_a, 2, false);

    float32_t data_b[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    uint64_t shape_b[] = {2, 5};
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    Tensor *c = tensor_matmul(a, b);
    TEST_ASSERT_EQUAL_UINT64(4, c->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(5, c->shape[1]);

    TEST_ASSERT_EQUAL_FLOAT(13.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(115.0f, c->data[19]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_reshape_to_1d(void) {
    float32_t data[] = {1, 2, 3, 4, 5, 6};
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_create(data, shape, 2, false);

    int64_t new_shape[] = {6};
    Tensor *reshaped = tensor_reshape(t, new_shape, 1);

    TEST_ASSERT_EQUAL_UINT64(1, reshaped->ndim);
    TEST_ASSERT_EQUAL_UINT64(6, reshaped->shape[0]);
    for (int i = 0; i < 6; i++) {
        TEST_ASSERT_EQUAL_FLOAT(data[i], reshaped->data[i]);
    }

    tensor_free(t);
    tensor_free(reshaped);
}

void test_tensor_reshape_with_minus_one_first(void) {
    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint64_t shape[] = {2, 4};
    Tensor *t = tensor_create(data, shape, 2, false);

    int64_t new_shape[] = {-1, 2};
    Tensor *reshaped = tensor_reshape(t, new_shape, 2);

    TEST_ASSERT_EQUAL_UINT64(4, reshaped->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, reshaped->shape[1]);

    tensor_free(t);
    tensor_free(reshaped);
}

void test_tensor_reshape_to_higher_dim(void) {
    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint64_t shape[] = {8};
    Tensor *t = tensor_create(data, shape, 1, false);

    int64_t new_shape[] = {2, 2, 2};
    Tensor *reshaped = tensor_reshape(t, new_shape, 3);

    TEST_ASSERT_EQUAL_UINT64(3, reshaped->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, reshaped->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, reshaped->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(2, reshaped->shape[2]);

    tensor_free(t);
    tensor_free(reshaped);
}

void test_tensor_reshape_preserves_data_order(void) {
    float32_t data[] = {1, 2, 3, 4, 5, 6};
    uint64_t shape[] = {1, 6};
    Tensor *t = tensor_create(data, shape, 2, false);

    int64_t new_shape[] = {3, 2};
    Tensor *reshaped = tensor_reshape(t, new_shape, 2);

    for (int i = 0; i < 6; i++) {
        TEST_ASSERT_EQUAL_FLOAT((float32_t)(i + 1), reshaped->data[i]);
    }

    tensor_free(t);
    tensor_free(reshaped);
}

void test_tensor_transpose_1d_noop(void) {
    float32_t data[] = {1, 2, 3, 4, 5};
    uint64_t shape[] = {5};
    Tensor *t = tensor_create(data, shape, 1, false);

    Tensor *t2 = tensor_transpose(t, 0, 0);
    TEST_ASSERT_EQUAL_UINT64(1, t2->ndim);
    for (int i = 0; i < 5; i++) {
        TEST_ASSERT_EQUAL_FLOAT(data[i], t2->data[i]);
    }

    tensor_free(t);
    tensor_free(t2);
}

void test_tensor_transpose_same_dims_noop(void) {
    float32_t data[] = {1, 2, 3, 4};
    uint64_t shape[] = {2, 2};
    Tensor *t = tensor_create(data, shape, 2, false);

    Tensor *t2 = tensor_transpose(t, 0, 0);
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_EQUAL_FLOAT(data[i], t2->data[i]);
    }

    tensor_free(t);
    tensor_free(t2);
}

void test_tensor_transpose_3d_middle_dims(void) {
    float32_t data[24];
    for (int i = 0; i < 24; i++)
        data[i] = (float32_t)i;
    uint64_t shape[] = {2, 3, 4};
    Tensor *t = tensor_create(data, shape, 3, false);

    Tensor *t2 = tensor_transpose(t, 1, 2);
    TEST_ASSERT_EQUAL_UINT64(2, t2->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(4, t2->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(3, t2->shape[2]);

    TEST_ASSERT_EQUAL_FLOAT(1.0f, t2->data[3]);

    tensor_free(t);
    tensor_free(t2);
}

void test_tensor_sum_negative_axis(void) {
    float32_t data[] = {1, 2, 3, 4, 5, 6};
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_create(data, shape, 2, false);

    Tensor *s = tensor_sum(t, -1, false);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, s->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(15.0f, s->data[1]);

    tensor_free(t);
    tensor_free(s);
}

void test_tensor_sum_reduce_to_scalar(void) {
    float32_t data[] = {1, 2, 3, 4, 5};
    uint64_t shape[] = {5};
    Tensor *t = tensor_create(data, shape, 1, false);

    Tensor *s = tensor_sum(t, 0, false);
    TEST_ASSERT_EQUAL_UINT64(0, s->ndim);
    TEST_ASSERT_EQUAL_FLOAT(15.0f, s->data[0]);

    tensor_free(t);
    tensor_free(s);
}

void test_tensor_mean_3d(void) {
    uint64_t shape[] = {2, 2, 3};
    Tensor *t = tensor_zeros(shape, 3, false);
    for (uint64_t i = 0; i < 12; i++) {
        t->data[i] = (float32_t)(i + 1);
    }

    Tensor *m = tensor_mean(t, 2, false);
    TEST_ASSERT_EQUAL_UINT64(2, m->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, m->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, m->shape[1]);

    TEST_ASSERT_EQUAL_FLOAT(2.0f, m->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(11.0f, m->data[3]);

    tensor_free(t);
    tensor_free(m);
}

void test_tensor_max_negative_values(void) {
    float32_t data[] = {-10, -5, -20, -1, -15, -8};
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_create(data, shape, 2, false);

    Tensor *m = tensor_max(t, 1, false);
    TEST_ASSERT_EQUAL_FLOAT(-5.0f, m->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(-1.0f, m->data[1]);

    tensor_free(t);
    tensor_free(m);
}

void test_tensor_max_keepdims_3d(void) {
    uint64_t shape[] = {2, 3, 4};
    Tensor *t = tensor_zeros(shape, 3, false);
    for (uint64_t i = 0; i < 24; i++) {
        t->data[i] = (float32_t)i;
    }

    Tensor *m = tensor_max(t, 1, true);
    TEST_ASSERT_EQUAL_UINT64(3, m->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, m->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(1, m->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(4, m->shape[2]);

    tensor_free(t);
    tensor_free(m);
}

void test_tensor_get_1d(void) {
    float32_t data[] = {10, 20, 30, 40, 50};
    uint64_t shape[] = {5};
    Tensor *t = tensor_create(data, shape, 1, false);

    uint64_t idx[] = {2};
    Tensor *val = tensor_get(t, idx);
    TEST_ASSERT_EQUAL_FLOAT(30.0f, val->data[0]);

    tensor_free(t);
    tensor_free(val);
}

void test_tensor_get_3d(void) {
    uint64_t shape[] = {2, 3, 4};
    Tensor *t = tensor_zeros(shape, 3, false);
    for (uint64_t i = 0; i < 24; i++) {
        t->data[i] = (float32_t)i;
    }

    uint64_t idx[] = {1, 2, 3};
    Tensor *val = tensor_get(t, idx);
    TEST_ASSERT_EQUAL_FLOAT(23.0f, val->data[0]);

    tensor_free(t);
    tensor_free(val);
}

void test_tensor_create_large(void) {
    uint64_t shape[] = {100, 100, 100};
    Tensor *t = tensor_zeros(shape, 3, false);
    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(1000000, t->size);
    tensor_free(t);
}

void test_tensor_create_scalar_zeros(void) {
    Tensor *t = tensor_zeros(NULL, 0, false);
    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(0, t->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, t->size);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, t->data[0]);
    tensor_free(t);
}

void test_tensor_create_odd_shape(void) {
    uint64_t shape[] = {1, 3, 1, 5, 1};
    Tensor *t = tensor_zeros(shape, 5, false);
    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(5, t->ndim);
    TEST_ASSERT_EQUAL_UINT64(15, t->size);
    tensor_free(t);
}

void test_tensor_create_scalar_val(void) {
    float32_t val = 123.456f;
    Tensor *t = tensor_create(&val, NULL, 0, false);
    TEST_ASSERT_EQUAL_FLOAT(123.456f, t->data[0]);
    tensor_free(t);
}

void test_tensor_zeros_large(void) {
    uint64_t shape[] = {1000, 1000};
    Tensor *t = tensor_zeros(shape, 2, false);
    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(1000000, t->size);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, t->data[999999]);
    tensor_free(t);
}

void test_tensor_add_negatives(void) {
    float32_t val_a = -5.0f;
    Tensor *a = tensor_create(&val_a, NULL, 0, false);
    float32_t val_b = -10.0f;
    Tensor *b = tensor_create(&val_b, NULL, 0, false);
    Tensor *c = tensor_add(a, b);
    TEST_ASSERT_EQUAL_FLOAT(-15.0f, c->data[0]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_sub_self_is_zero(void) {
    uint64_t shape[] = {2, 2};
    float32_t data[] = {1, 2, 3, 4};
    Tensor *a = tensor_create(data, shape, 2, false);
    Tensor *c = tensor_sub(a, a);
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_EQUAL_FLOAT(0.0f, c->data[i]);
    tensor_free(a);
    tensor_free(c);
}

void test_tensor_mul_by_zero(void) {
    float32_t val_a = 5.0f;
    Tensor *a = tensor_create(&val_a, NULL, 0, false);
    float32_t val_b = 0.0f;
    Tensor *b = tensor_create(&val_b, NULL, 0, false);
    Tensor *c = tensor_mul(a, b);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, c->data[0]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_div_by_one(void) {
    float32_t val_a = 17.0f;
    Tensor *a = tensor_create(&val_a, NULL, 0, false);
    float32_t val_b = 1.0f;
    Tensor *b = tensor_create(&val_b, NULL, 0, false);
    Tensor *c = tensor_div(a, b);
    TEST_ASSERT_EQUAL_FLOAT(17.0f, c->data[0]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_add_broadcast_3d_1d_leading(void) {
    uint64_t shape_a[] = {2, 2, 2};
    Tensor *a = tensor_zeros(shape_a, 3, false);
    float32_t data_b[] = {1, 2};
    uint64_t shape_b[] = {2};
    Tensor *b = tensor_create(data_b, shape_b, 1, false);

    Tensor *c = tensor_add(a, b);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, c->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, c->data[2]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_mul_broadcast_3d_2d_trailing(void) {
    uint64_t shape_a[] = {2, 3, 4};
    Tensor *a = tensor_zeros(shape_a, 3, false);
    for (int i = 0; i < 24; i++)
        a->data[i] = 1.0f;

    uint64_t shape_b[] = {3, 4};
    Tensor *b = tensor_zeros(shape_b, 2, false);
    for (int i = 0; i < 12; i++)
        b->data[i] = 2.0f;

    Tensor *c = tensor_mul(a, b);
    TEST_ASSERT_EQUAL_UINT64(3, c->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, c->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, c->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(4, c->shape[2]);

    TEST_ASSERT_EQUAL_FLOAT(2.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, c->data[23]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_sub_broadcast_scalar_negative(void) {
    uint64_t shape[] = {2, 2};
    Tensor *a = tensor_zeros(shape, 2, false);
    float32_t val = -5.0f;
    Tensor *b = tensor_create(&val, NULL, 0, false);

    Tensor *c = tensor_sub(a, b);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, c->data[0]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_div_broadcast_middle_dim(void) {
    uint64_t shape_a[] = {2, 3, 4};
    Tensor *a = tensor_zeros(shape_a, 3, false);
    for (int i = 0; i < 24; i++)
        a->data[i] = 10.0f;

    uint64_t shape_b[] = {3, 1};
    float32_t data_b[] = {2, 5, 2};
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    Tensor *c = tensor_div(a, b);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, c->data[4]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_complex_broadcast_shapes(void) {
    uint64_t shape_a[] = {1, 5, 1};
    Tensor *a = tensor_zeros(shape_a, 3, false);
    uint64_t shape_b[] = {4, 1, 6};
    Tensor *b = tensor_zeros(shape_b, 3, false);

    Tensor *c = tensor_add(a, b);
    TEST_ASSERT_EQUAL_UINT64(3, c->ndim);
    TEST_ASSERT_EQUAL_UINT64(4, c->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(5, c->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(6, c->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(120, c->size);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_arithmetic_chaining(void) {
    float32_t val1 = 1.0f;
    Tensor *t1 = tensor_create(&val1, NULL, 0, false);
    float32_t val2 = 2.0f;
    Tensor *t2 = tensor_create(&val2, NULL, 0, false);
    float32_t val3 = 3.0f;
    Tensor *t3 = tensor_create(&val3, NULL, 0, false);

    Tensor *tmp = tensor_add(t1, t2);
    Tensor *res = tensor_mul(tmp, t3);

    TEST_ASSERT_EQUAL_FLOAT(9.0f, res->data[0]);

    tensor_free(t1);
    tensor_free(t2);
    tensor_free(t3);
    tensor_free(tmp);
    tensor_free(res);
}

void test_tensor_matmul_zeros(void) {
    uint64_t shape[] = {2, 2};
    Tensor *a = tensor_zeros(shape, 2, false);
    Tensor *b = tensor_zeros(shape, 2, false);
    Tensor *c = tensor_matmul(a, b);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, c->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, c->data[3]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_matmul_negatives(void) {
    float32_t data_a[] = {-1, -1, -1, -1};
    uint64_t shape[] = {2, 2};
    Tensor *a = tensor_create(data_a, shape, 2, false);
    float32_t data_b[] = {1, 1, 1, 1};
    Tensor *b = tensor_create(data_b, shape, 2, false);
    Tensor *c = tensor_matmul(a, b);
    TEST_ASSERT_EQUAL_FLOAT(-2.0f, c->data[0]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_matmul_large_values(void) {
    float32_t data[] = {1000.0f, 1000.0f, 1000.0f, 1000.0f};
    uint64_t shape[] = {2, 2};
    Tensor *a = tensor_create(data, shape, 2, false);
    Tensor *b = tensor_create(data, shape, 2, false);
    Tensor *c = tensor_matmul(a, b);
    TEST_ASSERT_EQUAL_FLOAT(2000000.0f, c->data[0]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_matmul_transposed_inputs(void) {
    float32_t data_a[] = {1, 2, 3, 4, 5, 6};
    uint64_t shape_a[] = {2, 3};
    Tensor *a = tensor_create(data_a, shape_a, 2, false);

    Tensor *at = tensor_transpose(a, 0, 1);

    Tensor *c = tensor_matmul(at, a);
    TEST_ASSERT_EQUAL_UINT64(3, c->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, c->shape[1]);

    TEST_ASSERT_EQUAL_FLOAT(17.0f, c->data[0]);

    tensor_free(a);
    tensor_free(at);
    tensor_free(c);
}

void test_tensor_matmul_associativity_check(void) {
    float32_t data[] = {1, 2, 3, 4};
    uint64_t shape[] = {2, 2};
    Tensor *a = tensor_create(data, shape, 2, false);
    Tensor *b = tensor_create(data, shape, 2, false);
    Tensor *c = tensor_create(data, shape, 2, false);

    Tensor *ab = tensor_matmul(a, b);
    Tensor *res1 = tensor_matmul(ab, c);

    Tensor *bc = tensor_matmul(b, c);
    Tensor *res2 = tensor_matmul(a, bc);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_EQUAL_FLOAT(res1->data[i], res2->data[i]);
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(ab);
    tensor_free(res1);
    tensor_free(bc);
    tensor_free(res2);
}

void test_tensor_reshape_flatten_3d_to_1d(void) {
    uint64_t shape[] = {2, 2, 2};
    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor *t = tensor_create(data, shape, 3, false);
    int64_t new_shape[] = {-1};
    Tensor *r = tensor_reshape(t, new_shape, 1);
    TEST_ASSERT_EQUAL_UINT64(1, r->ndim);
    TEST_ASSERT_EQUAL_UINT64(8, r->shape[0]);
    TEST_ASSERT_EQUAL_FLOAT(8.0f, r->data[7]);
    tensor_free(t);
    tensor_free(r);
}

void test_tensor_reshape_expand_1d_to_3d(void) {
    uint64_t shape[] = {8};
    float32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor *t = tensor_create(data, shape, 1, false);
    int64_t new_shape[] = {2, 2, 2};
    Tensor *r = tensor_reshape(t, new_shape, 3);
    TEST_ASSERT_EQUAL_UINT64(3, r->ndim);
    TEST_ASSERT_EQUAL_FLOAT(8.0f, r->data[7]);
    tensor_free(t);
    tensor_free(r);
}

void test_tensor_reshape_auto_dim_middle(void) {
    uint64_t shape[] = {24};
    float32_t data[24] = {0};
    Tensor *t = tensor_create(data, shape, 1, false);
    int64_t new_shape[] = {2, -1, 4};
    Tensor *r = tensor_reshape(t, new_shape, 3);
    TEST_ASSERT_EQUAL_UINT64(3, r->shape[1]);
    tensor_free(t);
    tensor_free(r);
}

void test_tensor_transpose_double_swap(void) {
    uint64_t shape[] = {2, 3, 4};
    Tensor *t = tensor_zeros(shape, 3, false);
    Tensor *t1 = tensor_transpose(t, 0, 1);
    Tensor *t2 = tensor_transpose(t1, 1, 2);
    TEST_ASSERT_EQUAL_UINT64(3, t2->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(4, t2->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(2, t2->shape[2]);
    tensor_free(t);
    tensor_free(t1);
    tensor_free(t2);
}

void test_tensor_transpose_4d_dims(void) {
    uint64_t shape[] = {1, 2, 3, 4};
    Tensor *t = tensor_zeros(shape, 4, false);
    t->data[1] = 5.0f;

    Tensor *t2 = tensor_transpose(t, 1, 3);
    TEST_ASSERT_EQUAL_UINT64(1, t2->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(4, t2->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(3, t2->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(2, t2->shape[3]);

    tensor_free(t);
    tensor_free(t2);
}

void test_tensor_sum_keepdims_middle(void) {
    uint64_t shape[] = {2, 2, 2};
    float32_t data[] = {1, 1, 1, 1, 1, 1, 1, 1};
    Tensor *t = tensor_create(data, shape, 3, false);
    Tensor *s = tensor_sum(t, 1, true);
    TEST_ASSERT_EQUAL_UINT64(2, s->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(1, s->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(2, s->shape[2]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, s->data[0]);
    tensor_free(t);
    tensor_free(s);
}

void test_tensor_mean_negative_values(void) {
    float32_t data[] = {-2, -4};
    uint64_t shape[] = {2};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *m = tensor_mean(t, 0, false);
    TEST_ASSERT_EQUAL_FLOAT(-3.0f, m->data[0]);
    tensor_free(t);
    tensor_free(m);
}

void test_tensor_max_all_same(void) {
    float32_t data[] = {7, 7, 7};
    uint64_t shape[] = {3};
    Tensor *t = tensor_create(data, shape, 1, false);
    Tensor *m = tensor_max(t, 0, false);
    TEST_ASSERT_EQUAL_FLOAT(7.0f, m->data[0]);
    tensor_free(t);
    tensor_free(m);
}

void test_tensor_strides_check_after_transpose(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_zeros(shape, 2, false);
    Tensor *t2 = tensor_transpose(t, 0, 1);
    TEST_ASSERT_EQUAL_UINT64(2, t2->strides[0]);
    TEST_ASSERT_EQUAL_UINT64(1, t2->strides[1]);
    tensor_free(t);
    tensor_free(t2);
}

void test_tensor_get_boundary_values(void) {
    float32_t data[] = {1, 2, 3, 4};
    uint64_t shape[] = {4};
    Tensor *t = tensor_create(data, shape, 1, false);
    uint64_t idx[] = {3};
    Tensor *v = tensor_get(t, idx);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, v->data[0]);
    tensor_free(t);
    tensor_free(v);
}

void test_tensor_scalar_reduction(void) {
    float32_t val = 5.0f;
    uint64_t shape[] = {1};
    Tensor *t1 = tensor_create(&val, shape, 1, false);
    Tensor *s = tensor_sum(t1, 0, false);
    TEST_ASSERT_EQUAL_UINT64(0, s->ndim);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, s->data[0]);
    tensor_free(t1);
    tensor_free(s);
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
    RUN_TEST(test_tensor_requires_grad);
    RUN_TEST(test_tensor_broadcast_complex);
    RUN_TEST(test_tensor_transpose_general);
    RUN_TEST(test_tensor_div_broadcast);
    RUN_TEST(test_tensor_create_1d);
    RUN_TEST(test_tensor_create_3d);
    RUN_TEST(test_tensor_create_4d);
    RUN_TEST(test_tensor_create_single_element_non_scalar);
    RUN_TEST(test_tensor_zeros_initializes_correctly);
    RUN_TEST(test_tensor_strides_various_shapes);
    RUN_TEST(test_tensor_broadcast_scalar_to_tensor);
    RUN_TEST(test_tensor_broadcast_different_ndims);
    RUN_TEST(test_tensor_broadcast_multiple_1_dims);
    RUN_TEST(test_tensor_matmul_square);
    RUN_TEST(test_tensor_matmul_1x1);
    RUN_TEST(test_tensor_matmul_identity);
    RUN_TEST(test_tensor_matmul_tall_and_wide);
    RUN_TEST(test_tensor_reshape_to_1d);
    RUN_TEST(test_tensor_reshape_with_minus_one_first);
    RUN_TEST(test_tensor_reshape_to_higher_dim);
    RUN_TEST(test_tensor_reshape_preserves_data_order);
    RUN_TEST(test_tensor_transpose_1d_noop);
    RUN_TEST(test_tensor_transpose_same_dims_noop);
    RUN_TEST(test_tensor_transpose_3d_middle_dims);
    RUN_TEST(test_tensor_sum_negative_axis);
    RUN_TEST(test_tensor_sum_reduce_to_scalar);
    RUN_TEST(test_tensor_mean_3d);
    RUN_TEST(test_tensor_max_negative_values);
    RUN_TEST(test_tensor_max_keepdims_3d);
    RUN_TEST(test_tensor_get_1d);
    RUN_TEST(test_tensor_get_3d);
    RUN_TEST(test_tensor_create_large);
    RUN_TEST(test_tensor_create_scalar_zeros);
    RUN_TEST(test_tensor_create_odd_shape);
    RUN_TEST(test_tensor_create_scalar_val);
    RUN_TEST(test_tensor_zeros_large);
    RUN_TEST(test_tensor_add_negatives);
    RUN_TEST(test_tensor_sub_self_is_zero);
    RUN_TEST(test_tensor_mul_by_zero);
    RUN_TEST(test_tensor_div_by_one);
    RUN_TEST(test_tensor_add_broadcast_3d_1d_leading);
    RUN_TEST(test_tensor_mul_broadcast_3d_2d_trailing);
    RUN_TEST(test_tensor_sub_broadcast_scalar_negative);
    RUN_TEST(test_tensor_div_broadcast_middle_dim);
    RUN_TEST(test_tensor_complex_broadcast_shapes);
    RUN_TEST(test_tensor_arithmetic_chaining);
    RUN_TEST(test_tensor_matmul_zeros);
    RUN_TEST(test_tensor_matmul_negatives);
    RUN_TEST(test_tensor_matmul_large_values);
    RUN_TEST(test_tensor_matmul_transposed_inputs);
    RUN_TEST(test_tensor_matmul_associativity_check);
    RUN_TEST(test_tensor_reshape_flatten_3d_to_1d);
    RUN_TEST(test_tensor_reshape_expand_1d_to_3d);
    RUN_TEST(test_tensor_reshape_auto_dim_middle);
    RUN_TEST(test_tensor_transpose_double_swap);
    RUN_TEST(test_tensor_transpose_4d_dims);
    RUN_TEST(test_tensor_sum_keepdims_middle);
    RUN_TEST(test_tensor_mean_negative_values);
    RUN_TEST(test_tensor_max_all_same);
    RUN_TEST(test_tensor_strides_check_after_transpose);
    RUN_TEST(test_tensor_get_boundary_values);
    RUN_TEST(test_tensor_scalar_reduction);
    return UNITY_END();
}
