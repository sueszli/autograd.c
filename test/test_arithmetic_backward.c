#include "ops/arithmetic.h"
#include "ops/arithmetic_backward.h"
#include "tensor.h"
#include "unity.h"
#include <math.h>
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

static void test_add_backward(void) {
    uint64_t shape[] = {2};
    Tensor *a = tensor_create((float[]){1.0f, 2.0f}, shape, 1, true);
    Tensor *b = tensor_create((float[]){3.0f, 4.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float[]){1.0f, 1.0f}, shape, 1, false);

    Tensor *da = tensor_add_backward_a(grad, a);
    Tensor *db = tensor_add_backward_b(grad, b);

    TEST_ASSERT_NOT_NULL(da);
    TEST_ASSERT_EQUAL_UINT64(2, da->size);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, da->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, da->data[1]);

    TEST_ASSERT_NOT_NULL(db);
    TEST_ASSERT_EQUAL_UINT64(2, db->size);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, db->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, db->data[1]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(grad);
    tensor_free(da);
    tensor_free(db);
}

static void test_sub_backward(void) {
    uint64_t shape[] = {2};
    Tensor *a = tensor_create((float[]){1.0f, 2.0f}, shape, 1, true);
    Tensor *b = tensor_create((float[]){3.0f, 4.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float[]){1.0f, 2.0f}, shape, 1, false);

    Tensor *da = tensor_sub_backward_a(grad, a);
    Tensor *db = tensor_sub_backward_b(grad, b);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, da->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 2.0f, da->data[1]);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, -1.0f, db->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -2.0f, db->data[1]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(grad);
    tensor_free(da);
    tensor_free(db);
}

static void test_mul_backward(void) {
    uint64_t shape[] = {2};
    Tensor *a = tensor_create((float[]){2.0f, 3.0f}, shape, 1, true);
    Tensor *b = tensor_create((float[]){4.0f, 5.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float[]){1.0f, 1.0f}, shape, 1, false);

    Tensor *da = tensor_mul_backward_a(grad, a, b);
    Tensor *db = tensor_mul_backward_b(grad, a, b);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 4.0f, da->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 5.0f, da->data[1]);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 2.0f, db->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.0f, db->data[1]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(grad);
    tensor_free(da);
    tensor_free(db);
}

static void test_mul_backward_broadcast(void) {
    uint64_t shape_a[] = {2};
    uint64_t shape_b[] = {1};
    Tensor *a = tensor_create((float[]){1.0f, 2.0f}, shape_a, 1, true);
    Tensor *b = tensor_create((float[]){2.0f}, shape_b, 1, true);
    Tensor *grad = tensor_create((float[]){1.0f, 1.0f}, shape_a, 1, false);

    Tensor *da = tensor_mul_backward_a(grad, a, b);
    Tensor *db = tensor_mul_backward_b(grad, a, b);

    TEST_ASSERT_EQUAL_UINT64(1, da->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, da->shape[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 2.0f, da->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 2.0f, da->data[1]);

    TEST_ASSERT_EQUAL_UINT64(1, db->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, db->shape[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.0f, db->data[0]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(grad);
    tensor_free(da);
    tensor_free(db);
}

static void test_div_backward(void) {
    uint64_t shape[] = {2};
    Tensor *a = tensor_create((float[]){4.0f, 9.0f}, shape, 1, true);
    Tensor *b = tensor_create((float[]){2.0f, 3.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float[]){1.0f, 1.0f}, shape, 1, false);

    Tensor *da = tensor_div_backward_a(grad, a, b);
    Tensor *db = tensor_div_backward_b(grad, a, b);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.5f, da->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f / 3.0f, da->data[1]);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, -1.0f, db->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -1.0f, db->data[1]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(grad);
    tensor_free(da);
    tensor_free(db);
}

static void test_matmul_backward(void) {
    uint64_t shape_a[] = {2, 3};
    uint64_t shape_b[] = {3, 2};
    uint64_t shape_c[] = {2, 2};

    float data_a[] = {1, 2, 3, 4, 5, 6};
    float data_b[] = {7, 8, 9, 10, 11, 12};
    float data_grad[] = {1, 1, 1, 1};

    Tensor *a = tensor_create(data_a, shape_a, 2, true);
    Tensor *b = tensor_create(data_b, shape_b, 2, true);
    Tensor *grad = tensor_create(data_grad, shape_c, 2, false);

    Tensor *da = tensor_matmul_backward_a(grad, a, b);
    Tensor *db = tensor_matmul_backward_b(grad, a, b);

    TEST_ASSERT_EQUAL_UINT64(2, da->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, da->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, da->shape[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 15.0f, da->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 19.0f, da->data[1]);

    TEST_ASSERT_EQUAL_UINT64(2, db->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, db->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, db->shape[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 5.0f, db->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 7.0f, db->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 9.0f, db->data[4]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(grad);
    tensor_free(da);
    tensor_free(db);
}

static void test_add_backward_broadcast_2d(void) {
    uint64_t shape_a[] = {2, 2};
    uint64_t shape_b[] = {2, 1};
    Tensor *a = tensor_create((float[]){1, 2, 3, 4}, shape_a, 2, true);
    Tensor *b = tensor_create((float[]){10, 20}, shape_b, 2, true);
    Tensor *grad = tensor_create((float[]){1, 1, 1, 1}, shape_a, 2, false);

    Tensor *da = tensor_add_backward_a(grad, a);
    Tensor *db = tensor_add_backward_b(grad, b);

    TEST_ASSERT_EQUAL_UINT64(2, da->ndim);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, da->data[0]);

    TEST_ASSERT_EQUAL_UINT64(2, db->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, db->shape[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 2.0f, db->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 2.0f, db->data[1]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(grad);
    tensor_free(da);
    tensor_free(db);
}

static void test_sub_backward_broadcast(void) {
    uint64_t shape_a[] = {2};
    uint64_t shape_b[] = {1};
    Tensor *a = tensor_create((float[]){1.0f, 2.0f}, shape_a, 1, true);
    Tensor *b = tensor_create((float[]){10.0f}, shape_b, 1, true);
    Tensor *grad = tensor_create((float[]){1.0f, 1.0f}, shape_a, 1, false);

    Tensor *da = tensor_sub_backward_a(grad, a);
    Tensor *db = tensor_sub_backward_b(grad, b);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, da->data[0]);
    TEST_ASSERT_EQUAL_UINT64(1, db->shape[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -2.0f, db->data[0]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(grad);
    tensor_free(da);
    tensor_free(db);
}

static void test_mul_backward_zero(void) {
    uint64_t shape[] = {1};
    Tensor *a = tensor_create((float[]){5.0f}, shape, 1, true);
    Tensor *b = tensor_create((float[]){0.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float[]){2.0f}, shape, 1, false);

    Tensor *da = tensor_mul_backward_a(grad, a, b);
    Tensor *db = tensor_mul_backward_b(grad, a, b);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.0f, da->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 10.0f, db->data[0]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(grad);
    tensor_free(da);
    tensor_free(db);
}

static void test_div_backward_broadcast(void) {
    uint64_t shape_a[] = {2};
    uint64_t shape_b[] = {1};
    Tensor *a = tensor_create((float[]){10.0f, 20.0f}, shape_a, 1, true);
    Tensor *b = tensor_create((float[]){2.0f}, shape_b, 1, true);
    Tensor *grad = tensor_create((float[]){1.0f, 1.0f}, shape_a, 1, false);

    Tensor *da = tensor_div_backward_a(grad, a, b);

    Tensor *db = tensor_div_backward_b(grad, a, b);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.5f, da->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -7.5f, db->data[0]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(grad);
    tensor_free(da);
    tensor_free(db);
}

static void test_add_backward_scalar(void) {
    uint64_t shape[] = {1};
    Tensor *a = tensor_create((float[]){2.0f}, shape, 1, true);
    Tensor *b = tensor_create((float[]){3.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float[]){5.0f}, shape, 1, false);

    Tensor *da = tensor_add_backward_a(grad, a);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 5.0f, da->data[0]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(grad);
    tensor_free(da);
}

static void test_sub_backward_scalar(void) {
    uint64_t shape[] = {1};
    Tensor *a = tensor_create((float[]){2.0f}, shape, 1, true);
    Tensor *b = tensor_create((float[]){3.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float[]){5.0f}, shape, 1, false);

    Tensor *db = tensor_sub_backward_b(grad, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -5.0f, db->data[0]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(grad);
    tensor_free(db);
}

static void test_mul_backward_broadcast_result(void) {
    uint64_t shape_a[] = {2, 2, 2};
    uint64_t shape_b[] = {1};
    float data_a[] = {1, 1, 1, 1, 1, 1, 1, 1};
    Tensor *a = tensor_create(data_a, shape_a, 3, true);
    Tensor *b = tensor_create((float[]){2.0f}, shape_b, 1, true);
    Tensor *grad = tensor_create(data_a, shape_a, 3, false);

    Tensor *db = tensor_mul_backward_b(grad, a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 8.0f, db->data[0]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(grad);
    tensor_free(db);
}

static void test_square_via_mul(void) {
    uint64_t shape[] = {1};
    Tensor *x = tensor_create((float[]){3.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float[]){1.0f}, shape, 1, false);

    Tensor *d1 = tensor_mul_backward_a(grad, x, x);
    Tensor *d2 = tensor_mul_backward_b(grad, x, x);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.0f, d1->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.0f, d2->data[0]);

    tensor_free(x);
    tensor_free(grad);
    tensor_free(d1);
    tensor_free(d2);
}

static void test_neg_via_sub(void) {
    uint64_t shape[] = {1};
    Tensor *zero = tensor_create((float[]){0.0f}, shape, 1, true);
    Tensor *x = tensor_create((float[]){3.0f}, shape, 1, true);
    Tensor *grad = tensor_create((float[]){1.0f}, shape, 1, false);

    Tensor *dx = tensor_sub_backward_b(grad, x);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -1.0f, dx->data[0]);

    tensor_free(zero);
    tensor_free(x);
    tensor_free(grad);
    tensor_free(dx);
}

static void test_matmul_backward_broadcast_input(void) {
    uint64_t shape_a[] = {1, 2};
    uint64_t shape_b[] = {2, 3};
    uint64_t shape_c[] = {1, 3};
    Tensor *a = tensor_create((float[]){1, 2}, shape_a, 2, true);
    Tensor *b = tensor_create((float[]){1, 2, 3, 4, 5, 6}, shape_b, 2, true);
    Tensor *grad = tensor_create((float[]){1, 1, 1}, shape_c, 2, false);

    Tensor *da = tensor_matmul_backward_a(grad, a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 6.0f, da->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 15.0f, da->data[1]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(grad);
    tensor_free(da);
}

static void test_mul_large(void) {
    uint64_t shape[] = {100};
    Tensor *a = tensor_zeros(shape, 1, true);
    Tensor *b = tensor_zeros(shape, 1, true);
    for (int i = 0; i < 100; i++) {
        a->data[i] = (float)i;
        b->data[i] = 1.0f;
    }

    Tensor *grad = tensor_zeros(shape, 1, false);
    for (int i = 0; i < 100; i++) {
        grad->data[i] = 1.0f;
    }

    Tensor *da = tensor_mul_backward_a(grad, a, b);
    for (int i = 0; i < 100; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, da->data[i]);
    }

    tensor_free(a);
    tensor_free(b);
    tensor_free(grad);
    tensor_free(da);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_add_backward);
    RUN_TEST(test_sub_backward);
    RUN_TEST(test_mul_backward);
    RUN_TEST(test_mul_backward_broadcast);
    RUN_TEST(test_div_backward);
    RUN_TEST(test_matmul_backward);
    RUN_TEST(test_add_backward_broadcast_2d);
    RUN_TEST(test_sub_backward_broadcast);
    RUN_TEST(test_mul_backward_zero);
    RUN_TEST(test_div_backward_broadcast);
    RUN_TEST(test_add_backward_scalar);
    RUN_TEST(test_sub_backward_scalar);
    RUN_TEST(test_mul_backward_broadcast_result);
    RUN_TEST(test_square_via_mul);
    RUN_TEST(test_neg_via_sub);
    RUN_TEST(test_matmul_backward_broadcast_input);
    RUN_TEST(test_mul_large);
    return UNITY_END();
}
