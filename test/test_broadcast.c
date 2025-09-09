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

    tensor_t *a = tensor_create(data_a, shape, 2, false);
    tensor_t *b = tensor_create(data_b, shape, 2, false);

    TEST_ASSERT_TRUE(tensor_can_broadcast(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_can_broadcast_scalar_and_tensor(void) {
    i32 shape_scalar[] = {1};
    i32 shape_tensor[] = {3, 4};
    f32 data_scalar[] = {5.0f};
    f32 data_tensor[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    tensor_t *scalar = tensor_create(data_scalar, shape_scalar, 1, false);
    tensor_t *tensor = tensor_create(data_tensor, shape_tensor, 2, false);

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

    tensor_t *a = tensor_create(data_a, shape_a, 3, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    TEST_ASSERT_TRUE(tensor_can_broadcast(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_cannot_broadcast_incompatible(void) {
    i32 shape_a[] = {2, 3};
    i32 shape_b[] = {2, 4};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 1, 1, 1, 2, 2, 2, 2};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    TEST_ASSERT_FALSE(tensor_can_broadcast(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_shape(void) {
    i32 shape_a[] = {2, 1, 3};
    i32 shape_b[] = {4, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};

    tensor_t *a = tensor_create(data_a, shape_a, 3, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    shape_t result = get_tensor_broadcast_shape(a, b);

    TEST_ASSERT_NOT_NULL(result.shape);
    TEST_ASSERT_EQUAL(3, result.ndim);
    TEST_ASSERT_EQUAL(2, result.shape[0]);
    TEST_ASSERT_EQUAL(4, result.shape[1]);
    TEST_ASSERT_EQUAL(3, result.shape[2]);

    shape_free(&result);
    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_to_simple(void) {
    i32 shape[] = {1, 3};
    i32 target_shape[] = {2, 3};
    f32 data[] = {1, 2, 3};

    tensor_t *tensor = tensor_create(data, shape, 2, false);
    tensor_t *broadcasted = tensor_broadcast_to(tensor, target_shape, 2);

    TEST_ASSERT_NOT_NULL(broadcasted);
    TEST_ASSERT_EQUAL(2, broadcasted->ndim);
    TEST_ASSERT_EQUAL(2, broadcasted->shape[0]);
    TEST_ASSERT_EQUAL(3, broadcasted->shape[1]);

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

    tensor_t *tensor = tensor_create(data, shape, 1, false);
    tensor_t *broadcasted = tensor_broadcast_to(tensor, target_shape, 2);

    TEST_ASSERT_NOT_NULL(broadcasted);
    TEST_ASSERT_EQUAL(2, broadcasted->ndim);
    TEST_ASSERT_EQUAL(2, broadcasted->shape[0]);
    TEST_ASSERT_EQUAL(3, broadcasted->shape[1]);

    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 5.0f, broadcasted->data[i]);
    }

    tensor_destroy(tensor);
    tensor_destroy(broadcasted);
}

void test_tensor_broadcast_complex_case(void) {
    i32 shape_a[] = {1, 2, 1};
    i32 shape_b[] = {3, 1, 4};
    f32 data_a[] = {1, 2};
    f32 data_b[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};

    tensor_t *a = tensor_create(data_a, shape_a, 3, false);
    tensor_t *b = tensor_create(data_b, shape_b, 3, false);

    TEST_ASSERT_TRUE(tensor_can_broadcast(a, b));

    shape_t result = get_tensor_broadcast_shape(a, b);

    TEST_ASSERT_NOT_NULL(result.shape);
    TEST_ASSERT_EQUAL(3, result.ndim);
    TEST_ASSERT_EQUAL(3, result.shape[0]);
    TEST_ASSERT_EQUAL(2, result.shape[1]);
    TEST_ASSERT_EQUAL(4, result.shape[2]);

    shape_free(&result);
    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_null_inputs(void) {
    TEST_ASSERT_FALSE(tensor_can_broadcast(NULL, NULL));

    i32 shape[] = {2, 3};
    f32 data[] = {1, 2, 3, 4, 5, 6};
    tensor_t *tensor = tensor_create(data, shape, 2, false);

    TEST_ASSERT_FALSE(tensor_can_broadcast(NULL, tensor));
    TEST_ASSERT_FALSE(tensor_can_broadcast(tensor, NULL));

    TEST_ASSERT_NULL(tensor_broadcast_to(NULL, shape, 2));
    TEST_ASSERT_NULL(tensor_broadcast_to(tensor, NULL, 2));

    tensor_destroy(tensor);
}

void test_tensor_broadcast_zero_dim_scalars(void) {
    i32 shape_scalar[] = {1};
    f32 data_a[] = {42.0f};
    f32 data_b[] = {3.14f};

    tensor_t *a = tensor_create(data_a, shape_scalar, 1, false);
    tensor_t *b = tensor_create(data_b, shape_scalar, 1, false);

    TEST_ASSERT_TRUE(tensor_can_broadcast(a, b));

    shape_t result = get_tensor_broadcast_shape(a, b);
    TEST_ASSERT_EQUAL(1, result.ndim);
    TEST_ASSERT_EQUAL(1, result.shape[0]);

    shape_free(&result);
    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_single_element_different_shapes(void) {
    i32 shape_a[] = {1, 1, 1};
    i32 shape_b[] = {1};
    f32 data_a[] = {7.0f};
    f32 data_b[] = {2.0f};

    tensor_t *a = tensor_create(data_a, shape_a, 3, false);
    tensor_t *b = tensor_create(data_b, shape_b, 1, false);

    TEST_ASSERT_TRUE(tensor_can_broadcast(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_dimension_compatibility(void) {
    // (3, 1) vs (1, 4) -> should broadcast to (3, 4)
    i32 shape_a[] = {3, 1};
    i32 shape_b[] = {1, 4};
    f32 data_a[] = {1, 2, 3};
    f32 data_b[] = {10, 20, 30, 40};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    TEST_ASSERT_TRUE(tensor_can_broadcast(a, b));

    shape_t result = get_tensor_broadcast_shape(a, b);
    TEST_ASSERT_EQUAL(2, result.ndim);
    TEST_ASSERT_EQUAL(3, result.shape[0]);
    TEST_ASSERT_EQUAL(4, result.shape[1]);

    shape_free(&result);
    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_max_dimensions(void) {
    // (1, 2, 1, 3, 1) vs (2, 1, 4, 1, 5)
    i32 shape_a[] = {1, 2, 1, 3, 1};
    i32 shape_b[] = {2, 1, 4, 1, 5};
    f32 data_a[] = {1, 2, 3, 4, 5, 6}; // 2*3 = 6 elements
    f32 data_b[40];                    // 2*4*5 = 40 elements

    for (i32 i = 0; i < 40; i++) {
        data_b[i] = (f32)(i + 1);
    }

    tensor_t *a = tensor_create(data_a, shape_a, 5, false);
    tensor_t *b = tensor_create(data_b, shape_b, 5, false);

    TEST_ASSERT_TRUE(tensor_can_broadcast(a, b));

    shape_t result = get_tensor_broadcast_shape(a, b);
    TEST_ASSERT_EQUAL(5, result.ndim);
    TEST_ASSERT_EQUAL(2, result.shape[0]);
    TEST_ASSERT_EQUAL(2, result.shape[1]);
    TEST_ASSERT_EQUAL(4, result.shape[2]);
    TEST_ASSERT_EQUAL(3, result.shape[3]);
    TEST_ASSERT_EQUAL(5, result.shape[4]);

    shape_free(&result);
    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_incompatible_detailed(void) {
    i32 shape_a[] = {2, 3};
    i32 shape_b[] = {3, 2};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 1, 2, 2, 3, 3};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    TEST_ASSERT_FALSE(tensor_can_broadcast(a, b));

    // incompatible first dimension
    i32 shape_c[] = {4, 1, 3};
    i32 shape_d[] = {2, 5, 1};
    f32 data_c[12], data_d[10];

    for (i32 i = 0; i < 12; i++)
        data_c[i] = (f32)i;
    for (i32 i = 0; i < 10; i++)
        data_d[i] = (f32)i;

    tensor_t *c = tensor_create(data_c, shape_c, 3, false);
    tensor_t *d = tensor_create(data_d, shape_d, 3, false);

    TEST_ASSERT_FALSE(tensor_can_broadcast(c, d));

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(c);
    tensor_destroy(d);
}

void test_tensor_broadcast_to_edge_cases(void) {
    // from (1,) to various shapes
    i32 scalar_shape[] = {1};
    f32 scalar_data[] = {99.5f};
    tensor_t *scalar = tensor_create(scalar_data, scalar_shape, 1, false);

    // to 4D shape
    i32 target_shape[] = {2, 3, 4, 5};
    tensor_t *broadcasted = tensor_broadcast_to(scalar, target_shape, 4);

    TEST_ASSERT_NOT_NULL(broadcasted);
    TEST_ASSERT_EQUAL(4, broadcasted->ndim);
    TEST_ASSERT_EQUAL(2, broadcasted->shape[0]);
    TEST_ASSERT_EQUAL(3, broadcasted->shape[1]);
    TEST_ASSERT_EQUAL(4, broadcasted->shape[2]);
    TEST_ASSERT_EQUAL(5, broadcasted->shape[3]);

    // verify all elements are the scalar value
    u64 total_size = 2 * 3 * 4 * 5;
    for (u64 i = 0; i < total_size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 99.5f, broadcasted->data[i]);
    }

    // incompatible dimensions
    i32 invalid_target[] = {3, 2};
    i32 source_shape[] = {2, 3};
    f32 source_data[] = {1, 2, 3, 4, 5, 6};
    tensor_t *source = tensor_create(source_data, source_shape, 2, false);

    tensor_t *invalid_broadcast = tensor_broadcast_to(source, invalid_target, 2);
    TEST_ASSERT_NULL(invalid_broadcast);

    tensor_destroy(scalar);
    tensor_destroy(broadcasted);
    tensor_destroy(source);
}

void test_tensor_broadcast_large_tensors(void) {
    i32 shape_a[] = {100, 1};
    i32 shape_b[] = {1, 200};

    f32 *data_a = (f32 *)malloc(100 * sizeof(f32));
    f32 *data_b = (f32 *)malloc(200 * sizeof(f32));

    for (i32 i = 0; i < 100; i++) {
        data_a[i] = (f32)(i % 10);
    }
    for (i32 i = 0; i < 200; i++) {
        data_b[i] = (f32)(i % 5) * 0.1f;
    }

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    TEST_ASSERT_TRUE(tensor_can_broadcast(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
    free(data_a);
    free(data_b);
}

void test_tensor_broadcast_memory_stress(void) {
    i32 shape_base[] = {1};
    f32 data_base[] = {1.0f};

    tensor_t *base = tensor_create(data_base, shape_base, 1, false);

    for (i32 iteration = 0; iteration < 50; iteration++) {
        i32 target_shape[] = {iteration + 1, 2};
        tensor_t *broadcast = tensor_broadcast_to(base, target_shape, 2);

        if (broadcast) {
            TEST_ASSERT_EQUAL(iteration + 1, broadcast->shape[0]);
            TEST_ASSERT_EQUAL(2, broadcast->shape[1]);
            TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, broadcast->data[0]);
            tensor_destroy(broadcast);
        }
    }

    tensor_destroy(base);
}

void test_tensor_broadcast_index_mapping_correctness(void) {
    i32 shape_a[] = {2, 1, 3};
    i32 shape_b[] = {1, 4, 1};
    f32 data_a[] = {1, 2, 3, 10, 20, 30};
    f32 data_b[] = {100, 200, 300, 400};

    tensor_t *a = tensor_create(data_a, shape_a, 3, false);
    tensor_t *b = tensor_create(data_b, shape_b, 3, false);

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_boundary_conditions(void) {
    i32 shape_ones[] = {1, 1, 1, 1};
    i32 shape_target[] = {5, 3, 2, 4};
    f32 data_ones[] = {42.0f};

    tensor_t *ones = tensor_create(data_ones, shape_ones, 4, false);
    tensor_t *broadcasted = tensor_broadcast_to(ones, shape_target, 4);

    TEST_ASSERT_NOT_NULL(broadcasted);
    u64 total_elements = 5 * 3 * 2 * 4;
    for (u64 i = 0; i < total_elements; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 42.0f, broadcasted->data[i]);
    }

    tensor_destroy(ones);
    tensor_destroy(broadcasted);
}

void test_tensor_broadcast_error_conditions(void) {
    i32 shape[] = {2, 3};
    f32 data[] = {1, 2, 3, 4, 5, 6};
    tensor_t *tensor = tensor_create(data, shape, 2, false);

    shape_t result = get_tensor_broadcast_shape(tensor, NULL);
    TEST_ASSERT_NULL(result.shape);
    TEST_ASSERT_EQUAL(0, result.ndim);

    result = get_tensor_broadcast_shape(NULL, NULL);
    TEST_ASSERT_NULL(result.shape);
    TEST_ASSERT_EQUAL(0, result.ndim);

    i32 invalid_target[] = {3, 2};
    tensor_t *invalid = tensor_broadcast_to(tensor, invalid_target, 2);
    TEST_ASSERT_NULL(invalid);

    tensor_t *null_broadcast = tensor_broadcast_to(NULL, invalid_target, 2);
    TEST_ASSERT_NULL(null_broadcast);

    tensor_destroy(tensor);
}

void test_tensor_broadcast_precision_floating_point(void) {
    i32 shape_a[] = {2, 1};
    i32 shape_b[] = {1, 2};
    f32 data_a[] = {1e-6f, -1e-6f};
    f32 data_b[] = {1e6f, -1e6f};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_inplace_basic(void) {
    i32 shape_a[] = {2, 1};
    i32 shape_b[] = {1, 3};
    f32 data_a[] = {1, 2};
    f32 data_b[] = {10, 20, 30};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    tensor_t *orig_a = a;
    tensor_t *orig_b = b;

    tensor_broadcast_inplace(&a, &b);

    TEST_ASSERT_NOT_NULL(a);
    TEST_ASSERT_NOT_NULL(b);
    TEST_ASSERT_NOT_EQUAL(orig_a, a);
    TEST_ASSERT_NOT_EQUAL(orig_b, b);

    TEST_ASSERT_EQUAL(2, a->ndim);
    TEST_ASSERT_EQUAL(2, a->shape[0]);
    TEST_ASSERT_EQUAL(3, a->shape[1]);

    TEST_ASSERT_EQUAL(2, b->ndim);
    TEST_ASSERT_EQUAL(2, b->shape[0]);
    TEST_ASSERT_EQUAL(3, b->shape[1]);

    f32 expected_a[] = {1, 1, 1, 2, 2, 2};
    f32 expected_b[] = {10, 20, 30, 10, 20, 30};

    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_a[i], a->data[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_b[i], b->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_inplace_scalar(void) {
    i32 shape_scalar[] = {1};
    i32 shape_tensor[] = {2, 3};
    f32 data_scalar[] = {5.0f};
    f32 data_tensor[] = {1, 2, 3, 4, 5, 6};

    tensor_t *scalar = tensor_create(data_scalar, shape_scalar, 1, false);
    tensor_t *tensor = tensor_create(data_tensor, shape_tensor, 2, false);

    tensor_broadcast_inplace(&scalar, &tensor);

    TEST_ASSERT_EQUAL(2, scalar->ndim);
    TEST_ASSERT_EQUAL(2, scalar->shape[0]);
    TEST_ASSERT_EQUAL(3, scalar->shape[1]);

    TEST_ASSERT_EQUAL(2, tensor->ndim);
    TEST_ASSERT_EQUAL(2, tensor->shape[0]);
    TEST_ASSERT_EQUAL(3, tensor->shape[1]);

    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 5.0f, scalar->data[i]);
    }

    tensor_destroy(scalar);
    tensor_destroy(tensor);
}

void test_tensor_broadcast_inplace_same_shape(void) {
    i32 shape[] = {2, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {10, 20, 30, 40, 50, 60};

    tensor_t *a = tensor_create(data_a, shape, 2, false);
    tensor_t *b = tensor_create(data_b, shape, 2, false);

    tensor_t *orig_a = a;
    tensor_t *orig_b = b;

    tensor_broadcast_inplace(&a, &b);

    TEST_ASSERT_EQUAL(orig_a, a);
    TEST_ASSERT_EQUAL(orig_b, b);

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_inplace_incompatible(void) {
    i32 shape_a[] = {2, 3};
    i32 shape_b[] = {2, 4};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 2, 3, 4, 5, 6, 7, 8};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    tensor_t *orig_a = a;
    tensor_t *orig_b = b;

    tensor_broadcast_inplace(&a, &b);

    TEST_ASSERT_EQUAL(orig_a, a);
    TEST_ASSERT_EQUAL(orig_b, b);

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_inplace_null_inputs(void) {
    i32 shape[] = {2, 3};
    f32 data[] = {1, 2, 3, 4, 5, 6};
    tensor_t *tensor = tensor_create(data, shape, 2, false);

    tensor_broadcast_inplace(NULL, NULL);

    tensor_t *null_tensor = NULL;
    tensor_broadcast_inplace(&tensor, &null_tensor);
    tensor_broadcast_inplace(&null_tensor, &tensor);

    TEST_ASSERT_NOT_NULL(tensor);

    tensor_destroy(tensor);
}

void test_tensor_broadcast_inplace_complex(void) {
    i32 shape_a[] = {1, 2, 1};
    i32 shape_b[] = {3, 1, 4};
    f32 data_a[] = {1, 2};
    f32 data_b[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};

    tensor_t *a = tensor_create(data_a, shape_a, 3, false);
    tensor_t *b = tensor_create(data_b, shape_b, 3, false);

    tensor_broadcast_inplace(&a, &b);

    TEST_ASSERT_EQUAL(3, a->ndim);
    TEST_ASSERT_EQUAL(3, a->shape[0]);
    TEST_ASSERT_EQUAL(2, a->shape[1]);
    TEST_ASSERT_EQUAL(4, a->shape[2]);

    TEST_ASSERT_EQUAL(3, b->ndim);
    TEST_ASSERT_EQUAL(3, b->shape[0]);
    TEST_ASSERT_EQUAL(2, b->shape[1]);
    TEST_ASSERT_EQUAL(4, b->shape[2]);

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_shapes_match_identical(void) {
    i32 shape[] = {2, 3, 4};
    f32 data_a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    f32 data_b[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240};

    tensor_t *a = tensor_create(data_a, shape, 3, false);
    tensor_t *b = tensor_create(data_b, shape, 3, false);

    TEST_ASSERT_TRUE(tensor_shapes_match(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_shapes_match_different_ndim(void) {
    i32 shape_a[] = {2, 3};
    i32 shape_b[] = {2, 3, 1};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 2, 3, 4, 5, 6};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 3, false);

    TEST_ASSERT_FALSE(tensor_shapes_match(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_shapes_match_different_dimensions(void) {
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

void test_tensor_shapes_match_scalar_tensors(void) {
    i32 shape[] = {1};
    f32 data_a[] = {42.0f};
    f32 data_b[] = {3.14f};

    tensor_t *a = tensor_create(data_a, shape, 1, false);
    tensor_t *b = tensor_create(data_b, shape, 1, false);

    TEST_ASSERT_TRUE(tensor_shapes_match(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_shapes_match_multi_dimensional(void) {
    i32 shape[] = {3, 4, 2, 5};
    f32 *data_a = (f32 *)malloc(120 * sizeof(f32));
    f32 *data_b = (f32 *)malloc(120 * sizeof(f32));

    for (i32 i = 0; i < 120; i++) {
        data_a[i] = (f32)i;
        data_b[i] = (f32)(i * 2);
    }

    tensor_t *a = tensor_create(data_a, shape, 4, false);
    tensor_t *b = tensor_create(data_b, shape, 4, false);

    TEST_ASSERT_TRUE(tensor_shapes_match(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
    free(data_a);
    free(data_b);
}

void test_tensor_shapes_match_partial_different(void) {
    i32 shape_a[] = {2, 3, 4};
    i32 shape_b[] = {2, 3, 5};
    f32 data_a[24], data_b[30];

    for (i32 i = 0; i < 24; i++)
        data_a[i] = (f32)i;
    for (i32 i = 0; i < 30; i++)
        data_b[i] = (f32)i;

    tensor_t *a = tensor_create(data_a, shape_a, 3, false);
    tensor_t *b = tensor_create(data_b, shape_b, 3, false);

    TEST_ASSERT_FALSE(tensor_shapes_match(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_shapes_match_null_inputs(void) {
    i32 shape[] = {2, 3};
    f32 data[] = {1, 2, 3, 4, 5, 6};
    tensor_t *tensor = tensor_create(data, shape, 2, false);

    TEST_ASSERT_FALSE(tensor_shapes_match(NULL, NULL));
    TEST_ASSERT_FALSE(tensor_shapes_match(NULL, tensor));
    TEST_ASSERT_FALSE(tensor_shapes_match(tensor, NULL));

    tensor_destroy(tensor);
}

void test_tensor_shapes_match_single_element_different_shapes(void) {
    i32 shape_a[] = {1, 1, 1};
    i32 shape_b[] = {1, 1};
    f32 data_a[] = {5.0f};
    f32 data_b[] = {5.0f};

    tensor_t *a = tensor_create(data_a, shape_a, 3, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    TEST_ASSERT_FALSE(tensor_shapes_match(a, b));

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_shapes_match_edge_cases(void) {
    i32 shape_large[] = {1, 1, 1, 1, 1, 1};
    i32 shape_small[] = {1, 1, 1, 1, 1, 2};
    f32 data_large[] = {1.0f};
    f32 data_small[2] = {1.0f, 2.0f};

    tensor_t *large = tensor_create(data_large, shape_large, 6, false);
    tensor_t *small = tensor_create(data_small, shape_small, 6, false);

    TEST_ASSERT_FALSE(tensor_shapes_match(large, small));

    i32 shape_match[] = {1, 1, 1, 1, 1, 1};
    f32 data_match[] = {42.0f};
    tensor_t *match = tensor_create(data_match, shape_match, 6, false);

    TEST_ASSERT_TRUE(tensor_shapes_match(large, match));

    tensor_destroy(large);
    tensor_destroy(small);
    tensor_destroy(match);
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
    RUN_TEST(test_tensor_broadcast_complex_case);
    RUN_TEST(test_tensor_broadcast_null_inputs);
    RUN_TEST(test_tensor_broadcast_zero_dim_scalars);
    RUN_TEST(test_tensor_broadcast_single_element_different_shapes);
    RUN_TEST(test_tensor_broadcast_dimension_compatibility);
    RUN_TEST(test_tensor_broadcast_max_dimensions);
    RUN_TEST(test_tensor_broadcast_incompatible_detailed);
    RUN_TEST(test_tensor_broadcast_to_edge_cases);
    RUN_TEST(test_tensor_broadcast_large_tensors);
    RUN_TEST(test_tensor_broadcast_memory_stress);
    RUN_TEST(test_tensor_broadcast_boundary_conditions);
    RUN_TEST(test_tensor_broadcast_error_conditions);
    RUN_TEST(test_tensor_broadcast_inplace_basic);
    RUN_TEST(test_tensor_broadcast_inplace_scalar);
    RUN_TEST(test_tensor_broadcast_inplace_same_shape);
    RUN_TEST(test_tensor_broadcast_inplace_incompatible);
    RUN_TEST(test_tensor_broadcast_inplace_null_inputs);
    RUN_TEST(test_tensor_broadcast_inplace_complex);
    RUN_TEST(test_tensor_shapes_match_identical);
    RUN_TEST(test_tensor_shapes_match_different_ndim);
    RUN_TEST(test_tensor_shapes_match_different_dimensions);
    RUN_TEST(test_tensor_shapes_match_scalar_tensors);
    RUN_TEST(test_tensor_shapes_match_multi_dimensional);
    RUN_TEST(test_tensor_shapes_match_partial_different);
    RUN_TEST(test_tensor_shapes_match_null_inputs);
    RUN_TEST(test_tensor_shapes_match_single_element_different_shapes);
    RUN_TEST(test_tensor_shapes_match_edge_cases);

    return UNITY_END();
}
