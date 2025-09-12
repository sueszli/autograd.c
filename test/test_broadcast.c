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

void test_tensor_shapes_match_same_shape(void) {
    i32 shape[] = {2, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {6, 5, 4, 3, 2, 1};

    tensor_t *a = tensor_create(data_a, shape, 2, false);
    tensor_t *b = tensor_create(data_b, shape, 2, false);

    TEST_ASSERT_TRUE(tensor_shapes_match(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_shapes_match_different_shape(void) {
    i32 shape_a[] = {2, 3};
    i32 shape_b[] = {2, 4};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 2, 3, 4, 5, 6, 7, 8};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    TEST_ASSERT_FALSE(tensor_shapes_match(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_shapes_match_different_ndim(void) {
    i32 shape_a[] = {2, 3};
    i32 shape_b[] = {6};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 2, 3, 4, 5, 6};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 1, false);

    TEST_ASSERT_FALSE(tensor_shapes_match(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_reduce_broadcast_same_shape(void) {
    i32 shape[] = {2, 3};
    f32 data[] = {1, 2, 3, 4, 5, 6};

    tensor_t *broadcasted = tensor_create(data, shape, 2, false);
    tensor_t *target = tensor_create(data, shape, 2, false);

    tensor_t *result = tensor_reduce(broadcasted, target);

    TEST_ASSERT_EQUAL(2, result->ndim);
    TEST_ASSERT_EQUAL(2, result->shape[0]);
    TEST_ASSERT_EQUAL(3, result->shape[1]);

    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, data[i], result->data[i]);
    }

    tensor_destroy(broadcasted);
    tensor_destroy(target);
    tensor_destroy(result);
}

void test_tensor_reduce_broadcast_scalar_to_tensor(void) {
    i32 broadcasted_shape[] = {2, 3};
    i32 target_shape[] = {1};
    f32 broadcasted_data[] = {5, 5, 5, 5, 5, 5};
    f32 target_data[] = {0};

    tensor_t *broadcasted = tensor_create(broadcasted_data, broadcasted_shape, 2, false);
    tensor_t *target = tensor_create(target_data, target_shape, 1, false);

    tensor_t *result = tensor_reduce(broadcasted, target);

    TEST_ASSERT_EQUAL(1, result->ndim);
    TEST_ASSERT_EQUAL(1, result->shape[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 30.0f, result->data[0]);

    tensor_destroy(broadcasted);
    tensor_destroy(target);
    tensor_destroy(result);
}

void test_tensor_reduce_broadcast_dimension_reduction(void) {
    i32 broadcasted_shape[] = {2, 3};
    i32 target_shape[] = {1, 3};
    f32 broadcasted_data[] = {1, 2, 3, 4, 5, 6};
    f32 target_data[] = {0, 0, 0};

    tensor_t *broadcasted = tensor_create(broadcasted_data, broadcasted_shape, 2, false);
    tensor_t *target = tensor_create(target_data, target_shape, 2, false);

    tensor_t *result = tensor_reduce(broadcasted, target);

    TEST_ASSERT_EQUAL(2, result->ndim);
    TEST_ASSERT_EQUAL(1, result->shape[0]);
    TEST_ASSERT_EQUAL(3, result->shape[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 5.0f, result->data[0]); // 1 + 4
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 7.0f, result->data[1]); // 2 + 5
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 9.0f, result->data[2]); // 3 + 6

    tensor_destroy(broadcasted);
    tensor_destroy(target);
    tensor_destroy(result);
}

void test_tensor_reduce_broadcast_complex(void) {
    i32 broadcasted_shape[] = {2, 4, 3};
    i32 target_shape[] = {2, 1, 3};
    f32 broadcasted_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    f32 target_data[] = {0, 0, 0, 0, 0, 0};

    tensor_t *broadcasted = tensor_create(broadcasted_data, broadcasted_shape, 3, false);
    tensor_t *target = tensor_create(target_data, target_shape, 3, false);

    tensor_t *result = tensor_reduce(broadcasted, target);

    TEST_ASSERT_EQUAL(3, result->ndim);
    TEST_ASSERT_EQUAL(2, result->shape[0]);
    TEST_ASSERT_EQUAL(1, result->shape[1]);
    TEST_ASSERT_EQUAL(3, result->shape[2]);

    // expected: sum along the middle dimension (axis 1)
    // result[0, 0, :] = [1+4+7+10, 2+5+8+11, 3+6+9+12] = [22, 26, 30]
    // result[1, 0, :] = [13+16+19+22, 14+17+20+23, 15+18+21+24] = [70, 74, 78]
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 22.0f, result->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 26.0f, result->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 30.0f, result->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 70.0f, result->data[3]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 74.0f, result->data[4]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 78.0f, result->data[5]);

    tensor_destroy(broadcasted);
    tensor_destroy(target);
    tensor_destroy(result);
}

i32 main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_tensor_broadcast_same_shape);
    RUN_TEST(test_tensor_broadcast_scalar_and_tensor);
    RUN_TEST(test_tensor_broadcast_different_dimensions);
    RUN_TEST(test_tensor_broadcast_already_broadcasted);
    RUN_TEST(test_tensor_broadcast_1d_to_3d);
    RUN_TEST(test_tensor_broadcast_to_higher_dimension);

    RUN_TEST(test_tensor_shapes_match_same_shape);
    RUN_TEST(test_tensor_shapes_match_different_shape);
    RUN_TEST(test_tensor_shapes_match_different_ndim);

    RUN_TEST(test_tensor_reduce_broadcast_same_shape);
    RUN_TEST(test_tensor_reduce_broadcast_scalar_to_tensor);
    RUN_TEST(test_tensor_reduce_broadcast_dimension_reduction);
    RUN_TEST(test_tensor_reduce_broadcast_complex);

    return UNITY_END();
}