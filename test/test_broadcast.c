#include "../src/tensor/broadcast.h"
#include "../src/tensor/tensor.h"
#include "../src/utils/types.h"
#include <math.h>
#include <stdlib.h>
#include <unity.h>

void setUp(void) {}

void tearDown(void) {}

void test_tensor_can_broadcast_same_shape(void) {
    i32 shape[] = {2, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 1, 1, 2, 2, 2};

    Tensor *a = tensor_create(data_a, shape, 2, false);
    Tensor *b = tensor_create(data_b, shape, 2, false);

    TEST_ASSERT_TRUE(tensor_can_broadcast(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_can_broadcast_scalar_and_tensor(void) {
    i32 shape_scalar[] = {1};
    i32 shape_tensor[] = {3, 4};
    f32 data_scalar[] = {5.0f};
    f32 data_tensor[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    Tensor *scalar = tensor_create(data_scalar, shape_scalar, 1, false);
    Tensor *tensor = tensor_create(data_tensor, shape_tensor, 2, false);

    TEST_ASSERT_TRUE(tensor_can_broadcast(scalar, tensor));
    TEST_ASSERT_TRUE(tensor_can_broadcast(tensor, scalar));

    tensor_destroy(scalar);
    tensor_destroy(tensor);
}

void test_tensor_can_broadcast_different_dimensions(void) {
    i32 shape_a[] = {2, 1, 3};
    i32 shape_b[] = {4, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};

    Tensor *a = tensor_create(data_a, shape_a, 3, false);
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    TEST_ASSERT_TRUE(tensor_can_broadcast(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_cannot_broadcast_incompatible(void) {
    i32 shape_a[] = {2, 3};
    i32 shape_b[] = {2, 4};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 1, 1, 1, 2, 2, 2, 2};

    Tensor *a = tensor_create(data_a, shape_a, 2, false);
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    TEST_ASSERT_FALSE(tensor_can_broadcast(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_shape(void) {
    i32 shape_a[] = {2, 1, 3};
    i32 shape_b[] = {4, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};

    Tensor *a = tensor_create(data_a, shape_a, 3, false);
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    i32 result_ndim;
    i32 *result_shape = get_tensor_broadcast_shape(a, b, &result_ndim);

    TEST_ASSERT_NOT_NULL(result_shape);
    TEST_ASSERT_EQUAL(3, result_ndim);
    TEST_ASSERT_EQUAL(2, result_shape[0]);
    TEST_ASSERT_EQUAL(4, result_shape[1]);
    TEST_ASSERT_EQUAL(3, result_shape[2]);

    free(result_shape);
    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_to_simple(void) {
    i32 shape[] = {1, 3};
    i32 target_shape[] = {2, 3};
    f32 data[] = {1, 2, 3};

    Tensor *tensor = tensor_create(data, shape, 2, false);
    Tensor *broadcasted = tensor_broadcast_to(tensor, target_shape, 2);

    TEST_ASSERT_NOT_NULL(broadcasted);
    TEST_ASSERT_EQUAL(2, broadcasted->ndim);
    TEST_ASSERT_EQUAL(2, broadcasted->shape[0]);
    TEST_ASSERT_EQUAL(3, broadcasted->shape[1]);

    // check data is correctly broadcasted
    f32 expected[] = {1, 2, 3, 1, 2, 3};
    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], broadcasted->data[i]);
    }

    tensor_destroy(tensor);
    tensor_destroy(broadcasted);
}

void test_tensor_broadcast_to_scalar(void) {
    i32 shape[] = {1};
    i32 target_shape[] = {2, 3};
    f32 data[] = {5.0f};

    Tensor *tensor = tensor_create(data, shape, 1, false);
    Tensor *broadcasted = tensor_broadcast_to(tensor, target_shape, 2);

    TEST_ASSERT_NOT_NULL(broadcasted);
    TEST_ASSERT_EQUAL(2, broadcasted->ndim);
    TEST_ASSERT_EQUAL(2, broadcasted->shape[0]);
    TEST_ASSERT_EQUAL(3, broadcasted->shape[1]);

    // check all values are 5.0
    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 5.0f, broadcasted->data[i]);
    }

    tensor_destroy(tensor);
    tensor_destroy(broadcasted);
}

void test_tensor_add_broadcast_simple(void) {
    i32 shape_a[] = {2, 1};
    i32 shape_b[] = {3};
    f32 data_a[] = {1, 4};
    f32 data_b[] = {10, 20, 30};

    Tensor *a = tensor_create(data_a, shape_a, 2, false);
    Tensor *b = tensor_create(data_b, shape_b, 1, false);

    Tensor *result = tensor_add_broadcast(a, b);

    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(2, result->ndim);
    TEST_ASSERT_EQUAL(2, result->shape[0]);
    TEST_ASSERT_EQUAL(3, result->shape[1]);

    // [[1+10, 1+20, 1+30], [4+10, 4+20, 4+30]]
    f32 expected[] = {11, 21, 31, 14, 24, 34};
    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_mul_broadcast_simple(void) {
    i32 shape_a[] = {2, 1};
    i32 shape_b[] = {3};
    f32 data_a[] = {2, 3};
    f32 data_b[] = {10, 20, 30};

    Tensor *a = tensor_create(data_a, shape_a, 2, false);
    Tensor *b = tensor_create(data_b, shape_b, 1, false);

    Tensor *result = tensor_mul_broadcast(a, b);

    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(2, result->ndim);
    TEST_ASSERT_EQUAL(2, result->shape[0]);
    TEST_ASSERT_EQUAL(3, result->shape[1]);

    // [[2*10, 2*20, 2*30], [3*10, 3*20, 3*30]]
    f32 expected[] = {20, 40, 60, 30, 60, 90};
    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_broadcast_complex_case(void) {
    i32 shape_a[] = {1, 2, 1};
    i32 shape_b[] = {3, 1, 4};
    f32 data_a[] = {1, 2};
    f32 data_b[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};

    Tensor *a = tensor_create(data_a, shape_a, 3, false);
    Tensor *b = tensor_create(data_b, shape_b, 3, false);

    TEST_ASSERT_TRUE(tensor_can_broadcast(a, b));

    i32 result_ndim;
    i32 *result_shape = get_tensor_broadcast_shape(a, b, &result_ndim);

    TEST_ASSERT_NOT_NULL(result_shape);
    TEST_ASSERT_EQUAL(3, result_ndim);
    TEST_ASSERT_EQUAL(3, result_shape[0]);
    TEST_ASSERT_EQUAL(2, result_shape[1]);
    TEST_ASSERT_EQUAL(4, result_shape[2]);

    Tensor *result = tensor_add_broadcast(a, b);
    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(3, result->ndim);
    TEST_ASSERT_EQUAL(3, result->shape[0]);
    TEST_ASSERT_EQUAL(2, result->shape[1]);
    TEST_ASSERT_EQUAL(4, result->shape[2]);

    free(result_shape);
    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_broadcast_null_inputs(void) {
    TEST_ASSERT_FALSE(tensor_can_broadcast(NULL, NULL));

    i32 shape[] = {2, 3};
    f32 data[] = {1, 2, 3, 4, 5, 6};
    Tensor *tensor = tensor_create(data, shape, 2, false);

    TEST_ASSERT_FALSE(tensor_can_broadcast(NULL, tensor));
    TEST_ASSERT_FALSE(tensor_can_broadcast(tensor, NULL));

    TEST_ASSERT_NULL(tensor_broadcast_to(NULL, shape, 2));
    TEST_ASSERT_NULL(tensor_broadcast_to(tensor, NULL, 2));

    TEST_ASSERT_NULL(tensor_add_broadcast(NULL, tensor));
    TEST_ASSERT_NULL(tensor_add_broadcast(tensor, NULL));

    TEST_ASSERT_NULL(tensor_mul_broadcast(NULL, tensor));
    TEST_ASSERT_NULL(tensor_mul_broadcast(tensor, NULL));

    tensor_destroy(tensor);
}

i32 main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_tensor_can_broadcast_same_shape);
    RUN_TEST(test_tensor_can_broadcast_scalar_and_tensor);
    RUN_TEST(test_tensor_can_broadcast_different_dimensions);
    RUN_TEST(test_tensor_cannot_broadcast_incompatible);
    RUN_TEST(test_tensor_broadcast_shape);
    RUN_TEST(test_tensor_broadcast_to_simple);
    RUN_TEST(test_tensor_broadcast_to_scalar);
    RUN_TEST(test_tensor_add_broadcast_simple);
    RUN_TEST(test_tensor_mul_broadcast_simple);
    RUN_TEST(test_tensor_broadcast_complex_case);
    RUN_TEST(test_tensor_broadcast_null_inputs);

    return UNITY_END();
}