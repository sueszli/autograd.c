#include "../src/tensor/ops.h"
#include "../src/tensor/tensor.h"
#include "../src/utils/types.h"
#include <math.h>
#include <stdlib.h>
#include <unity.h>

void setUp(void) {}

void tearDown(void) {}

void test_tensor_add_same_shape(void) {
    i32 shape[] = {2, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 1, 1, 2, 2, 2};
    f32 expected[] = {2, 3, 4, 6, 7, 8};

    tensor_t *a = tensor_create(data_a, shape, 2, false);
    tensor_t *b = tensor_create(data_b, shape, 2, false);

    tensor_t *result = tensor_add(a, b);

    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(2, result->ndim);
    TEST_ASSERT_EQUAL(2, result->shape[0]);
    TEST_ASSERT_EQUAL(3, result->shape[1]);

    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_add_with_broadcasting(void) {
    i32 shape_a[] = {2, 3};
    i32 shape_b[] = {1, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {10, 20, 30};
    f32 expected[] = {11, 22, 33, 14, 25, 36};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    tensor_t *result = tensor_add(a, b);

    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(2, result->ndim);
    TEST_ASSERT_EQUAL(2, result->shape[0]);
    TEST_ASSERT_EQUAL(3, result->shape[1]);

    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_sub_same_shape(void) {
    i32 shape[] = {2, 2};
    f32 data_a[] = {5, 6, 7, 8};
    f32 data_b[] = {1, 2, 3, 4};
    f32 expected[] = {4, 4, 4, 4};

    tensor_t *a = tensor_create(data_a, shape, 2, false);
    tensor_t *b = tensor_create(data_b, shape, 2, false);

    tensor_t *result = tensor_sub(a, b);

    TEST_ASSERT_NOT_NULL(result);
    for (i32 i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_mul_with_broadcasting(void) {
    i32 shape_a[] = {2, 1};
    i32 shape_b[] = {1, 3};
    f32 data_a[] = {2, 3};
    f32 data_b[] = {4, 5, 6};
    f32 expected[] = {8, 10, 12, 12, 15, 18};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    tensor_t *result = tensor_mul(a, b);

    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(2, result->ndim);
    TEST_ASSERT_EQUAL(2, result->shape[0]);
    TEST_ASSERT_EQUAL(3, result->shape[1]);

    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_div_same_shape(void) {
    i32 shape[] = {2, 2};
    f32 data_a[] = {8, 12, 16, 20};
    f32 data_b[] = {2, 3, 4, 5};
    f32 expected[] = {4, 4, 4, 4};

    tensor_t *a = tensor_create(data_a, shape, 2, false);
    tensor_t *b = tensor_create(data_b, shape, 2, false);

    tensor_t *result = tensor_div(a, b);

    TEST_ASSERT_NOT_NULL(result);
    for (i32 i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_scalar_broadcasting(void) {
    i32 shape_tensor[] = {2, 3};
    i32 shape_scalar[] = {1};
    f32 data_tensor[] = {1, 2, 3, 4, 5, 6};
    f32 data_scalar[] = {10};
    f32 expected_add[] = {11, 12, 13, 14, 15, 16};
    f32 expected_mul[] = {10, 20, 30, 40, 50, 60};

    tensor_t *tensor = tensor_create(data_tensor, shape_tensor, 2, false);
    tensor_t *scalar = tensor_create(data_scalar, shape_scalar, 1, false);

    tensor_t *result_add = tensor_add(tensor, scalar);
    TEST_ASSERT_NOT_NULL(result_add);
    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_add[i], result_add->data[i]);
    }

    tensor_t *result_mul = tensor_mul(tensor, scalar);
    TEST_ASSERT_NOT_NULL(result_mul);
    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_mul[i], result_mul->data[i]);
    }

    tensor_destroy(tensor);
    tensor_destroy(scalar);
    tensor_destroy(result_add);
    tensor_destroy(result_mul);
}

void test_tensor_add_gradient(void) {
    i32 shape[] = {2, 2};
    f32 data_a[] = {1, 2, 3, 4};
    f32 data_b[] = {2, 3, 4, 5};

    tensor_t *a = tensor_create(data_a, shape, 2, true);
    tensor_t *b = tensor_create(data_b, shape, 2, true);

    tensor_t *result = tensor_add(a, b);
    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_TRUE(result->requires_grad);
    TEST_ASSERT_NOT_NULL(result->grad_fn);

    result->grad = tensor_create(NULL, result->shape, result->ndim, false);
    for (i32 i = 0; i < 4; i++) {
        result->grad->data[i] = 1.0f;
    }

    result->grad_fn(result);

    TEST_ASSERT_NOT_NULL(a->grad);
    TEST_ASSERT_NOT_NULL(b->grad);
    for (i32 i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, a->grad->data[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, b->grad->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_add_broadcast_grad(void) {
    i32 shape_a[] = {2, 1};
    i32 shape_b[] = {1, 3};
    f32 data_a[] = {1, 2};
    f32 data_b[] = {10, 20, 30};

    tensor_t *a = tensor_create(data_a, shape_a, 2, true);
    tensor_t *b = tensor_create(data_b, shape_b, 2, true);

    tensor_t *result = tensor_add(a, b);
    result->grad = tensor_create(NULL, result->shape, result->ndim, false);
    for (i32 i = 0; i < 6; i++) {
        result->grad->data[i] = 1.0f;
    }

    result->grad_fn(result);

    TEST_ASSERT_NOT_NULL(a->grad);
    TEST_ASSERT_EQUAL(2, a->grad->ndim);
    TEST_ASSERT_EQUAL(2, a->grad->shape[0]);
    TEST_ASSERT_EQUAL(1, a->grad->shape[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 3.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 3.0f, a->grad->data[1]);

    TEST_ASSERT_NOT_NULL(b->grad);
    TEST_ASSERT_EQUAL(2, b->grad->ndim);
    TEST_ASSERT_EQUAL(1, b->grad->shape[0]);
    TEST_ASSERT_EQUAL(3, b->grad->shape[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 2.0f, b->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 2.0f, b->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 2.0f, b->grad->data[2]);

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_div_broadcast_grad(void) {
    i32 shape_a[] = {2, 1};
    i32 shape_b[] = {1};
    f32 data_a[] = {10, 20};
    f32 data_b[] = {2};

    tensor_t *a = tensor_create(data_a, shape_a, 2, true);
    tensor_t *b = tensor_create(data_b, shape_b, 1, true);

    tensor_t *result = tensor_div(a, b);
    result->grad = tensor_create(NULL, result->shape, result->ndim, false);
    for (i32 i = 0; i < 2; i++) {
        result->grad->data[i] = 1.0f;
    }

    result->grad_fn(result);

    TEST_ASSERT_NOT_NULL(a->grad);
    TEST_ASSERT_EQUAL(2, a->grad->ndim);
    TEST_ASSERT_EQUAL(2, a->grad->shape[0]);
    TEST_ASSERT_EQUAL(1, a->grad->shape[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.5f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.5f, a->grad->data[1]);

    TEST_ASSERT_NOT_NULL(b->grad);
    TEST_ASSERT_EQUAL(1, b->grad->ndim);
    TEST_ASSERT_EQUAL(1, b->grad->shape[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, -7.5f, b->grad->data[0]);

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

i32 main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_tensor_add_same_shape);
    RUN_TEST(test_tensor_add_with_broadcasting);
    RUN_TEST(test_tensor_sub_same_shape);
    RUN_TEST(test_tensor_mul_with_broadcasting);
    RUN_TEST(test_tensor_div_same_shape);
    RUN_TEST(test_tensor_scalar_broadcasting);
    RUN_TEST(test_tensor_add_gradient);
    RUN_TEST(test_tensor_add_broadcast_grad);
    RUN_TEST(test_tensor_div_broadcast_grad);

    return UNITY_END();
}