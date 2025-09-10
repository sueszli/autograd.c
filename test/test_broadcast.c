#include "../src/tensor/broadcast.h"
#include "../src/tensor/tensor.h"
#include "../src/utils/types.h"
#include <math.h>
#include <stdlib.h>
#include <unity.h>

void setUp(void) {}

void tearDown(void) {}

void test_tensor_broadcast_same_shape(void) {
    i32 shape[] = {2, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 1, 1, 2, 2, 2};

    tensor_t *a = tensor_create(data_a, shape, 2, false);
    tensor_t *b = tensor_create(data_b, shape, 2, false);

    broadcasted_tensors_t result = tensor_broadcast(a, b);

    TEST_ASSERT_EQUAL_PTR(a, result.a);
    TEST_ASSERT_EQUAL_PTR(b, result.b);

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_scalar_and_tensor(void) {
    i32 shape_scalar[] = {1};
    i32 shape_tensor[] = {2, 3};
    f32 data_scalar[] = {5.0f};
    f32 data_tensor[] = {1, 2, 3, 4, 5, 6};

    tensor_t *scalar = tensor_create(data_scalar, shape_scalar, 1, false);
    tensor_t *tensor = tensor_create(data_tensor, shape_tensor, 2, false);

    broadcasted_tensors_t result = tensor_broadcast(scalar, tensor);

    TEST_ASSERT_NOT_EQUAL(scalar, result.a);
    TEST_ASSERT_NOT_EQUAL(tensor, result.b);

    TEST_ASSERT_EQUAL(2, result.a->ndim);
    TEST_ASSERT_EQUAL(2, result.a->shape[0]);
    TEST_ASSERT_EQUAL(3, result.a->shape[1]);

    TEST_ASSERT_EQUAL(2, result.b->ndim);
    TEST_ASSERT_EQUAL(2, result.b->shape[0]);
    TEST_ASSERT_EQUAL(3, result.b->shape[1]);

    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 5.0f, result.a->data[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-6, data_tensor[i], result.b->data[i]);
    }

    tensor_destroy(scalar);
    tensor_destroy(tensor);
    tensor_destroy(result.a);
    tensor_destroy(result.b);
}

void test_tensor_broadcast_different_dimensions(void) {
    i32 shape_a[] = {2, 1, 3};
    i32 shape_b[] = {4, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};

    tensor_t *a = tensor_create(data_a, shape_a, 3, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    broadcasted_tensors_t result = tensor_broadcast(a, b);

    TEST_ASSERT_EQUAL(3, result.a->ndim);
    TEST_ASSERT_EQUAL(2, result.a->shape[0]);
    TEST_ASSERT_EQUAL(4, result.a->shape[1]);
    TEST_ASSERT_EQUAL(3, result.a->shape[2]);

    TEST_ASSERT_EQUAL(3, result.b->ndim);
    TEST_ASSERT_EQUAL(2, result.b->shape[0]);
    TEST_ASSERT_EQUAL(4, result.b->shape[1]);
    TEST_ASSERT_EQUAL(3, result.b->shape[2]);

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result.a);
    tensor_destroy(result.b);
}

void test_tensor_broadcast_already_broadcasted(void) {
    i32 shape_a[] = {2, 3};
    i32 shape_b[] = {2, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 1, 1, 2, 2, 2};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    broadcasted_tensors_t result = tensor_broadcast(a, b);

    TEST_ASSERT_EQUAL_PTR(a, result.a);
    TEST_ASSERT_EQUAL_PTR(b, result.b);

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_1d_to_3d(void) {
    i32 shape_a[] = {3};
    i32 shape_b[] = {2, 1, 3};
    f32 data_a[] = {1, 2, 3};
    f32 data_b[] = {10, 20, 30, 40, 50, 60};

    tensor_t *a = tensor_create(data_a, shape_a, 1, false);
    tensor_t *b = tensor_create(data_b, shape_b, 3, false);

    broadcasted_tensors_t result = tensor_broadcast(a, b);

    TEST_ASSERT_EQUAL(3, result.a->ndim);
    TEST_ASSERT_EQUAL(2, result.a->shape[0]);
    TEST_ASSERT_EQUAL(1, result.a->shape[1]);
    TEST_ASSERT_EQUAL(3, result.a->shape[2]);

    TEST_ASSERT_EQUAL(3, result.b->ndim);
    TEST_ASSERT_EQUAL(2, result.b->shape[0]);
    TEST_ASSERT_EQUAL(1, result.b->shape[1]);
    TEST_ASSERT_EQUAL(3, result.b->shape[2]);

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result.a);
    tensor_destroy(result.b);
}

void test_tensor_broadcast_to_higher_dimension(void) {
    i32 shape_a[] = {2, 1};
    i32 shape_b[] = {1, 3, 2, 1};
    f32 data_a[] = {1, 2};
    f32 data_b[] = {10, 20, 30, 40, 50, 60};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 4, false);

    broadcasted_tensors_t result = tensor_broadcast(a, b);

    TEST_ASSERT_EQUAL(4, result.a->ndim);
    TEST_ASSERT_EQUAL(1, result.a->shape[0]);
    TEST_ASSERT_EQUAL(3, result.a->shape[1]);
    TEST_ASSERT_EQUAL(2, result.a->shape[2]);
    TEST_ASSERT_EQUAL(1, result.a->shape[3]);

    TEST_ASSERT_EQUAL(4, result.b->ndim);

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result.a);
    tensor_destroy(result.b);
}

i32 main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_tensor_broadcast_same_shape);
    RUN_TEST(test_tensor_broadcast_scalar_and_tensor);
    RUN_TEST(test_tensor_broadcast_different_dimensions);
    RUN_TEST(test_tensor_broadcast_already_broadcasted);
    RUN_TEST(test_tensor_broadcast_1d_to_3d);
    RUN_TEST(test_tensor_broadcast_to_higher_dimension);

    return UNITY_END();
}