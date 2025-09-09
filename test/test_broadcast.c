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

void test_tensor_broadcast_zero_dim_scalars(void) {
    i32 shape_scalar[] = {1};
    f32 data_a[] = {42.0f};
    f32 data_b[] = {3.14f};

    Tensor *a = tensor_create(data_a, shape_scalar, 1, false);
    Tensor *b = tensor_create(data_b, shape_scalar, 1, false);

    TEST_ASSERT_TRUE(tensor_can_broadcast(a, b));

    i32 result_ndim;
    i32 *result_shape = get_tensor_broadcast_shape(a, b, &result_ndim);
    TEST_ASSERT_EQUAL(1, result_ndim);
    TEST_ASSERT_EQUAL(1, result_shape[0]);

    Tensor *result = tensor_add_broadcast(a, b);
    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 45.14f, result->data[0]);

    free(result_shape);
    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_broadcast_single_element_different_shapes(void) {
    i32 shape_a[] = {1, 1, 1};
    i32 shape_b[] = {1};
    f32 data_a[] = {7.0f};
    f32 data_b[] = {2.0f};

    Tensor *a = tensor_create(data_a, shape_a, 3, false);
    Tensor *b = tensor_create(data_b, shape_b, 1, false);

    TEST_ASSERT_TRUE(tensor_can_broadcast(a, b));

    Tensor *result = tensor_mul_broadcast(a, b);
    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(3, result->ndim);
    TEST_ASSERT_EQUAL(1, result->shape[0]);
    TEST_ASSERT_EQUAL(1, result->shape[1]);
    TEST_ASSERT_EQUAL(1, result->shape[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 14.0f, result->data[0]);

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_broadcast_dimension_compatibility(void) {
    // (3, 1) vs (1, 4) -> should broadcast to (3, 4)
    i32 shape_a[] = {3, 1};
    i32 shape_b[] = {1, 4};
    f32 data_a[] = {1, 2, 3};
    f32 data_b[] = {10, 20, 30, 40};

    Tensor *a = tensor_create(data_a, shape_a, 2, false);
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    TEST_ASSERT_TRUE(tensor_can_broadcast(a, b));

    i32 result_ndim;
    i32 *result_shape = get_tensor_broadcast_shape(a, b, &result_ndim);
    TEST_ASSERT_EQUAL(2, result_ndim);
    TEST_ASSERT_EQUAL(3, result_shape[0]);
    TEST_ASSERT_EQUAL(4, result_shape[1]);

    Tensor *result = tensor_add_broadcast(a, b);
    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(2, result->ndim);
    TEST_ASSERT_EQUAL(3, result->shape[0]);
    TEST_ASSERT_EQUAL(4, result->shape[1]);

    // [1+10, 1+20, 1+30, 1+40, 2+10, 2+20, 2+30, 2+40, 3+10, 3+20, 3+30, 3+40]
    f32 expected[] = {11, 21, 31, 41, 12, 22, 32, 42, 13, 23, 33, 43};
    for (i32 i = 0; i < 12; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    free(result_shape);
    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_broadcast_max_dimensions(void) {
    // (1, 2, 1, 3, 1) vs (2, 1, 4, 1, 5)
    i32 shape_a[] = {1, 2, 1, 3, 1};
    i32 shape_b[] = {2, 1, 4, 1, 5};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};  // 2*3 = 6 elements
    f32 data_b[40];  // 2*4*5 = 40 elements
    
    for (i32 i = 0; i < 40; i++) {
        data_b[i] = (f32)(i + 1);
    }

    Tensor *a = tensor_create(data_a, shape_a, 5, false);
    Tensor *b = tensor_create(data_b, shape_b, 5, false);

    TEST_ASSERT_TRUE(tensor_can_broadcast(a, b));

    i32 result_ndim;
    i32 *result_shape = get_tensor_broadcast_shape(a, b, &result_ndim);
    TEST_ASSERT_EQUAL(5, result_ndim);
    TEST_ASSERT_EQUAL(2, result_shape[0]);
    TEST_ASSERT_EQUAL(2, result_shape[1]);
    TEST_ASSERT_EQUAL(4, result_shape[2]);
    TEST_ASSERT_EQUAL(3, result_shape[3]);
    TEST_ASSERT_EQUAL(5, result_shape[4]);

    free(result_shape);
    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_broadcast_incompatible_detailed(void) {
    i32 shape_a[] = {2, 3};
    i32 shape_b[] = {3, 2};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 1, 2, 2, 3, 3};

    Tensor *a = tensor_create(data_a, shape_a, 2, false);
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    TEST_ASSERT_FALSE(tensor_can_broadcast(a, b));
    TEST_ASSERT_NULL(tensor_add_broadcast(a, b));

    // incompatible first dimension
    i32 shape_c[] = {4, 1, 3};
    i32 shape_d[] = {2, 5, 1};
    f32 data_c[12], data_d[10];
    
    for (i32 i = 0; i < 12; i++) data_c[i] = (f32)i;
    for (i32 i = 0; i < 10; i++) data_d[i] = (f32)i;

    Tensor *c = tensor_create(data_c, shape_c, 3, false);
    Tensor *d = tensor_create(data_d, shape_d, 3, false);

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
    Tensor *scalar = tensor_create(scalar_data, scalar_shape, 1, false);

    // to 4D shape
    i32 target_shape[] = {2, 3, 4, 5};
    Tensor *broadcasted = tensor_broadcast_to(scalar, target_shape, 4);
    
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
    Tensor *source = tensor_create(source_data, source_shape, 2, false);
    
    Tensor *invalid_broadcast = tensor_broadcast_to(source, invalid_target, 2);
    TEST_ASSERT_NULL(invalid_broadcast);

    tensor_destroy(scalar);
    tensor_destroy(broadcasted);
    tensor_destroy(source);
}

void test_tensor_broadcast_large_tensors(void) {
    i32 shape_a[] = {100, 1};
    i32 shape_b[] = {1, 200};
    
    f32 *data_a = (f32*)malloc(100 * sizeof(f32));
    f32 *data_b = (f32*)malloc(200 * sizeof(f32));
    
    for (i32 i = 0; i < 100; i++) {
        data_a[i] = (f32)(i % 10);
    }
    for (i32 i = 0; i < 200; i++) {
        data_b[i] = (f32)(i % 5) * 0.1f;
    }
    
    Tensor *a = tensor_create(data_a, shape_a, 2, false);
    Tensor *b = tensor_create(data_b, shape_b, 2, false);
    
    TEST_ASSERT_TRUE(tensor_can_broadcast(a, b));
    
    Tensor *result = tensor_add_broadcast(a, b);
    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(2, result->ndim);
    TEST_ASSERT_EQUAL(100, result->shape[0]);
    TEST_ASSERT_EQUAL(200, result->shape[1]);
    
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, result->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.1f, result->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, result->data[200]);
    
    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
    free(data_a);
    free(data_b);
}

void test_tensor_broadcast_memory_stress(void) {
    i32 shape_base[] = {1};
    f32 data_base[] = {1.0f};
    
    Tensor *base = tensor_create(data_base, shape_base, 1, false);
    
    for (i32 iteration = 0; iteration < 50; iteration++) {
        i32 target_shape[] = {iteration + 1, 2};
        Tensor *broadcast = tensor_broadcast_to(base, target_shape, 2);
        
        if (broadcast) {
            TEST_ASSERT_EQUAL(iteration + 1, broadcast->shape[0]);
            TEST_ASSERT_EQUAL(2, broadcast->shape[1]);
            TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, broadcast->data[0]);
            tensor_destroy(broadcast);
        }
    }
    
    tensor_destroy(base);
}

void test_tensor_broadcast_mathematical_properties(void) {
    i32 shape_a[] = {2, 1};
    i32 shape_b[] = {1, 3};
    f32 data_a[] = {2.0f, -1.5f};
    f32 data_b[] = {3.0f, 0.0f, -2.0f};
    
    Tensor *a = tensor_create(data_a, shape_a, 2, false);
    Tensor *b = tensor_create(data_b, shape_b, 2, false);
    
    Tensor *add_result = tensor_add_broadcast(a, b);
    Tensor *mul_result = tensor_mul_broadcast(a, b);
    
    TEST_ASSERT_NOT_NULL(add_result);
    TEST_ASSERT_NOT_NULL(mul_result);
    
    Tensor *add_commute = tensor_add_broadcast(b, a);
    TEST_ASSERT_NOT_NULL(add_commute);
    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, add_result->data[i], add_commute->data[i]);
    }
    
    Tensor *mul_commute = tensor_mul_broadcast(b, a);
    TEST_ASSERT_NOT_NULL(mul_commute);
    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, mul_result->data[i], mul_commute->data[i]);
    }
    
    f32 expected_add[] = {5.0f, 2.0f, 0.0f, 1.5f, -1.5f, -3.5f};
    f32 expected_mul[] = {6.0f, 0.0f, -4.0f, -4.5f, 0.0f, 3.0f};
    
    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_add[i], add_result->data[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_mul[i], mul_result->data[i]);
    }
    
    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(add_result);
    tensor_destroy(mul_result);
    tensor_destroy(add_commute);
    tensor_destroy(mul_commute);
}

void test_tensor_broadcast_index_mapping_correctness(void) {
    i32 shape_a[] = {2, 1, 3};
    i32 shape_b[] = {1, 4, 1};
    f32 data_a[] = {1, 2, 3, 10, 20, 30};
    f32 data_b[] = {100, 200, 300, 400};
    
    Tensor *a = tensor_create(data_a, shape_a, 3, false);
    Tensor *b = tensor_create(data_b, shape_b, 3, false);
    
    Tensor *result = tensor_add_broadcast(a, b);
    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(3, result->ndim);
    TEST_ASSERT_EQUAL(2, result->shape[0]);
    TEST_ASSERT_EQUAL(4, result->shape[1]);
    TEST_ASSERT_EQUAL(3, result->shape[2]);
    
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 101.0f, result->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 102.0f, result->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 201.0f, result->data[3]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 110.0f, result->data[12]);
    
    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_broadcast_boundary_conditions(void) {
    i32 shape_ones[] = {1, 1, 1, 1};
    i32 shape_target[] = {5, 3, 2, 4};
    f32 data_ones[] = {42.0f};
    
    Tensor *ones = tensor_create(data_ones, shape_ones, 4, false);
    Tensor *broadcasted = tensor_broadcast_to(ones, shape_target, 4);
    
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
    Tensor *tensor = tensor_create(data, shape, 2, false);
    
    // Test with NULL tensor parameter
    i32 result_ndim;
    i32 *result_shape = get_tensor_broadcast_shape(tensor, NULL, &result_ndim);
    TEST_ASSERT_NULL(result_shape);
    
    // Test with both tensors NULL
    result_shape = get_tensor_broadcast_shape(NULL, NULL, &result_ndim);
    TEST_ASSERT_NULL(result_shape);
    
    // Test invalid broadcast_to with incompatible dimensions
    i32 invalid_target[] = {3, 2};
    Tensor *invalid = tensor_broadcast_to(tensor, invalid_target, 2);
    TEST_ASSERT_NULL(invalid);
    
    // Test broadcast_to with NULL tensor
    Tensor *null_broadcast = tensor_broadcast_to(NULL, invalid_target, 2);
    TEST_ASSERT_NULL(null_broadcast);
    
    tensor_destroy(tensor);
}

void test_tensor_broadcast_precision_floating_point(void) {
    i32 shape_a[] = {2, 1};
    i32 shape_b[] = {1, 2};
    f32 data_a[] = {1e-6f, -1e-6f};
    f32 data_b[] = {1e6f, -1e6f};
    
    Tensor *a = tensor_create(data_a, shape_a, 2, false);
    Tensor *b = tensor_create(data_b, shape_b, 2, false);
    
    Tensor *result = tensor_add_broadcast(a, b);
    TEST_ASSERT_NOT_NULL(result);
    
    TEST_ASSERT_FLOAT_WITHIN(1e-3, 1e6f, result->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-3, -1e6f, result->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-3, 1e6f, result->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-3, -1e6f, result->data[3]);
    
    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
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
    RUN_TEST(test_tensor_broadcast_zero_dim_scalars);
    RUN_TEST(test_tensor_broadcast_single_element_different_shapes);
    RUN_TEST(test_tensor_broadcast_dimension_compatibility);
    RUN_TEST(test_tensor_broadcast_max_dimensions);
    RUN_TEST(test_tensor_broadcast_incompatible_detailed);
    RUN_TEST(test_tensor_broadcast_to_edge_cases);
    RUN_TEST(test_tensor_broadcast_large_tensors);
    RUN_TEST(test_tensor_broadcast_memory_stress);
    RUN_TEST(test_tensor_broadcast_mathematical_properties);
    RUN_TEST(test_tensor_broadcast_index_mapping_correctness);
    RUN_TEST(test_tensor_broadcast_boundary_conditions);
    RUN_TEST(test_tensor_broadcast_error_conditions);
    RUN_TEST(test_tensor_broadcast_precision_floating_point);

    return UNITY_END();
}
