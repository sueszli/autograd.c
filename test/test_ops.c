#include "../src/tensor/ops.h"
#include "../src/tensor/tensor.h"
#include "../src/utils/types.h"
#include <math.h>
#include <stdlib.h>
#include <unity.h>

void setUp(void) {}

void tearDown(void) {}

void test_tensor_op_add_same_shape_no_broadcast(void) {
    i32 shape[] = {2, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 1, 1, 2, 2, 2};
    f32 expected[] = {2, 3, 4, 6, 7, 8};

    tensor_t *a = tensor_create(data_a, shape, 2, false);
    tensor_t *b = tensor_create(data_b, shape, 2, false);

    tensor_t *result = tensor_op_add(a, b, false);

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

void test_tensor_op_add_with_broadcasting(void) {
    i32 shape_a[] = {2, 3};
    i32 shape_b[] = {1, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {10, 20, 30};
    f32 expected[] = {11, 22, 33, 14, 25, 36};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    tensor_t *result = tensor_op_add(a, b, true);

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

void test_tensor_op_sub_same_shape(void) {
    i32 shape[] = {2, 2};
    f32 data_a[] = {5, 6, 7, 8};
    f32 data_b[] = {1, 2, 3, 4};
    f32 expected[] = {4, 4, 4, 4};

    tensor_t *a = tensor_create(data_a, shape, 2, false);
    tensor_t *b = tensor_create(data_b, shape, 2, false);

    tensor_t *result = tensor_op_sub(a, b, false);

    TEST_ASSERT_NOT_NULL(result);
    for (i32 i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_op_mul_with_broadcasting(void) {
    i32 shape_a[] = {2, 1};
    i32 shape_b[] = {1, 3};
    f32 data_a[] = {2, 3};
    f32 data_b[] = {4, 5, 6};
    f32 expected[] = {8, 10, 12, 12, 15, 18};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    tensor_t *result = tensor_op_mul(a, b, true);

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

void test_tensor_op_div_same_shape(void) {
    i32 shape[] = {2, 2};
    f32 data_a[] = {8, 12, 16, 20};
    f32 data_b[] = {2, 3, 4, 5};
    f32 expected[] = {4, 4, 4, 4};

    tensor_t *a = tensor_create(data_a, shape, 2, false);
    tensor_t *b = tensor_create(data_b, shape, 2, false);

    tensor_t *result = tensor_op_div(a, b, false);

    TEST_ASSERT_NOT_NULL(result);
    for (i32 i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_op_div_by_zero(void) {
    i32 shape[] = {2, 2};
    f32 data_a[] = {1, 2, 3, 4};
    f32 data_b[] = {1, 0, 2, 3};

    tensor_t *a = tensor_create(data_a, shape, 2, false);
    tensor_t *b = tensor_create(data_b, shape, 2, false);

    tensor_t *result = tensor_op_div(a, b, false);

    TEST_ASSERT_NULL(result);

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_op_incompatible_shapes_no_broadcast(void) {
    i32 shape_a[] = {2, 3};
    i32 shape_b[] = {2, 4};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 1, 1, 1, 2, 2, 2, 2};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    tensor_t *result = tensor_op_add(a, b, false);
    TEST_ASSERT_NULL(result);

    result = tensor_op_mul(a, b, true);
    TEST_ASSERT_NULL(result);

    tensor_destroy(a);
    tensor_destroy(b);
}

void test_tensor_op_scalar_broadcasting(void) {
    i32 shape_tensor[] = {2, 3};
    i32 shape_scalar[] = {1};
    f32 data_tensor[] = {1, 2, 3, 4, 5, 6};
    f32 data_scalar[] = {10};
    f32 expected_add[] = {11, 12, 13, 14, 15, 16};
    f32 expected_mul[] = {10, 20, 30, 40, 50, 60};

    tensor_t *tensor = tensor_create(data_tensor, shape_tensor, 2, false);
    tensor_t *scalar = tensor_create(data_scalar, shape_scalar, 1, false);

    tensor_t *result_add = tensor_op_add(tensor, scalar, true);
    TEST_ASSERT_NOT_NULL(result_add);
    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_add[i], result_add->data[i]);
    }

    tensor_t *result_mul = tensor_op_mul(tensor, scalar, true);
    TEST_ASSERT_NOT_NULL(result_mul);
    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_mul[i], result_mul->data[i]);
    }

    tensor_destroy(tensor);
    tensor_destroy(scalar);
    tensor_destroy(result_add);
    tensor_destroy(result_mul);
}

void test_tensor_op_generic_function(void) {
    i32 shape[] = {2, 2};
    f32 data_a[] = {1, 2, 3, 4};
    f32 data_b[] = {5, 6, 7, 8};

    tensor_t *a = tensor_create(data_a, shape, 2, false);
    tensor_t *b = tensor_create(data_b, shape, 2, false);

    tensor_t *result_add = tensor_op_generic(a, b, TENSOR_OP_ADD, false);
    TEST_ASSERT_NOT_NULL(result_add);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 6.0f, result_add->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 8.0f, result_add->data[1]);

    tensor_t *result_mul = tensor_op_generic(a, b, TENSOR_OP_MUL, false);
    TEST_ASSERT_NOT_NULL(result_mul);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 5.0f, result_mul->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 12.0f, result_mul->data[1]);

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result_add);
    tensor_destroy(result_mul);
}

void test_tensor_op_gradient_computation(void) {
    i32 shape[] = {2, 2};
    f32 data_a[] = {1, 2, 3, 4};
    f32 data_b[] = {2, 3, 4, 5};

    tensor_t *a = tensor_create(data_a, shape, 2, true);
    tensor_t *b = tensor_create(data_b, shape, 2, true);

    tensor_t *result = tensor_op_add(a, b, false);
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

void test_tensor_op_null_inputs(void) {
    i32 shape[] = {2, 2};
    f32 data[] = {1, 2, 3, 4};
    tensor_t *tensor = tensor_create(data, shape, 2, false);

    TEST_ASSERT_NULL(tensor_op_add(NULL, tensor, false));
    TEST_ASSERT_NULL(tensor_op_add(tensor, NULL, false));
    TEST_ASSERT_NULL(tensor_op_add(NULL, NULL, false));

    TEST_ASSERT_NULL(tensor_op_mul(NULL, tensor, true));
    TEST_ASSERT_NULL(tensor_op_sub(tensor, NULL, true));
    TEST_ASSERT_NULL(tensor_op_div(NULL, NULL, true));

    tensor_destroy(tensor);
}

void test_tensor_op_complex_broadcasting(void) {
    i32 shape_a[] = {2, 1, 3};
    i32 shape_b[] = {4, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    f32 data_b[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};

    tensor_t *a = tensor_create(data_a, shape_a, 3, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    tensor_t *result = tensor_op_add(a, b, true);

    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(3, result->ndim);
    TEST_ASSERT_EQUAL(2, result->shape[0]);
    TEST_ASSERT_EQUAL(4, result->shape[1]);
    TEST_ASSERT_EQUAL(3, result->shape[2]);

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_op_edge_cases(void) {
    i32 shape_single[] = {1};
    f32 data_single[] = {5.0f};

    tensor_t *single = tensor_create(data_single, shape_single, 1, false);

    tensor_t *result = tensor_op_add(single, single, false);
    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 10.0f, result->data[0]);

    tensor_destroy(single);
    tensor_destroy(result);
}

void test_tensor_op_large_tensors(void) {
    i32 shape[] = {100, 100};
    f32 *data_a = (f32 *)malloc(10000 * sizeof(f32));
    f32 *data_b = (f32 *)malloc(10000 * sizeof(f32));

    for (i32 i = 0; i < 10000; i++) {
        data_a[i] = (f32)i;
        data_b[i] = 1.0f;
    }

    tensor_t *a = tensor_create(data_a, shape, 2, false);
    tensor_t *b = tensor_create(data_b, shape, 2, false);

    tensor_t *result = tensor_op_add(a, b, false);
    TEST_ASSERT_NOT_NULL(result);

    for (i32 i = 0; i < 100; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, (f32)i + 1.0f, result->data[i]);
    }

    free(data_a);
    free(data_b);
    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_op_gradient_backward_functions(void) {
    i32 shape[] = {2, 2};
    f32 data_a[] = {2, 3, 4, 5};
    f32 data_b[] = {1, 2, 1, 2};

    tensor_t *a = tensor_create(data_a, shape, 2, true);
    tensor_t *b = tensor_create(data_b, shape, 2, true);

    tensor_t *result_mul = tensor_op_mul(a, b, false);
    TEST_ASSERT_NOT_NULL(result_mul);
    TEST_ASSERT_TRUE(result_mul->requires_grad);

    result_mul->grad = tensor_create(NULL, result_mul->shape, result_mul->ndim, false);
    for (i32 i = 0; i < 4; i++) {
        result_mul->grad->data[i] = 1.0f;
    }

    result_mul->grad_fn(result_mul);

    TEST_ASSERT_NOT_NULL(a->grad);
    TEST_ASSERT_NOT_NULL(b->grad);

    for (i32 i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, data_b[i], a->grad->data[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-6, data_a[i], b->grad->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result_mul);
}

void test_tensor_op_subtract_gradients(void) {
    i32 shape[] = {2, 2};
    f32 data_a[] = {5, 6, 7, 8};
    f32 data_b[] = {1, 2, 3, 4};

    tensor_t *a = tensor_create(data_a, shape, 2, true);
    tensor_t *b = tensor_create(data_b, shape, 2, true);

    tensor_t *result = tensor_op_sub(a, b, false);
    TEST_ASSERT_NOT_NULL(result);

    result->grad = tensor_create(NULL, result->shape, result->ndim, false);
    for (i32 i = 0; i < 4; i++) {
        result->grad->data[i] = 1.0f;
    }

    result->grad_fn(result);

    for (i32 i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, a->grad->data[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-6, -1.0f, b->grad->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_op_division_gradients(void) {
    i32 shape[] = {2, 2};
    f32 data_a[] = {8, 12, 16, 20};
    f32 data_b[] = {2, 3, 4, 5};

    tensor_t *a = tensor_create(data_a, shape, 2, true);
    tensor_t *b = tensor_create(data_b, shape, 2, true);

    tensor_t *result = tensor_op_div(a, b, false);
    TEST_ASSERT_NOT_NULL(result);

    result->grad = tensor_create(NULL, result->shape, result->ndim, false);
    for (i32 i = 0; i < 4; i++) {
        result->grad->data[i] = 1.0f;
    }

    result->grad_fn(result);

    for (i32 i = 0; i < 4; i++) {
        f32 expected_a_grad = 1.0f / data_b[i];
        f32 expected_b_grad = -data_a[i] / (data_b[i] * data_b[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_a_grad, a->grad->data[i]);
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_b_grad, b->grad->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_op_broadcasting_gradients(void) {
    i32 shape_a[] = {2, 1};
    i32 shape_b[] = {1, 3};
    f32 data_a[] = {2, 3};
    f32 data_b[] = {4, 5, 6};

    tensor_t *a = tensor_create(data_a, shape_a, 2, true);
    tensor_t *b = tensor_create(data_b, shape_b, 2, true);

    tensor_t *result = tensor_op_mul(a, b, true);
    TEST_ASSERT_NOT_NULL(result);

    result->grad = tensor_create(NULL, result->shape, result->ndim, false);
    for (i32 i = 0; i < 6; i++) {
        result->grad->data[i] = 1.0f;
    }

    result->grad_fn(result);

    TEST_ASSERT_NOT_NULL(a->grad);
    TEST_ASSERT_NOT_NULL(b->grad);

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_op_memory_management(void) {
    i32 shape[] = {3, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    f32 data_b[] = {9, 8, 7, 6, 5, 4, 3, 2, 1};

    for (i32 iter = 0; iter < 100; iter++) {
        tensor_t *a = tensor_create(data_a, shape, 2, false);
        tensor_t *b = tensor_create(data_b, shape, 2, false);

        tensor_t *result1 = tensor_op_add(a, b, false);
        tensor_t *result2 = tensor_op_mul(a, b, false);
        tensor_t *result3 = tensor_op_sub(a, b, false);
        tensor_t *result4 = tensor_op_div(a, b, false);

        TEST_ASSERT_NOT_NULL(result1);
        TEST_ASSERT_NOT_NULL(result2);
        TEST_ASSERT_NOT_NULL(result3);
        TEST_ASSERT_NOT_NULL(result4);

        tensor_destroy(a);
        tensor_destroy(b);
        tensor_destroy(result1);
        tensor_destroy(result2);
        tensor_destroy(result3);
        tensor_destroy(result4);
    }
}

void test_tensor_op_chained_operations(void) {
    i32 shape[] = {2, 2};
    f32 data_a[] = {1, 2, 3, 4};
    f32 data_b[] = {2, 2, 2, 2};
    f32 data_c[] = {3, 3, 3, 3};

    tensor_t *a = tensor_create(data_a, shape, 2, false);
    tensor_t *b = tensor_create(data_b, shape, 2, false);
    tensor_t *c = tensor_create(data_c, shape, 2, false);

    tensor_t *temp = tensor_op_add(a, b, false);
    tensor_t *result = tensor_op_mul(temp, c, false);

    TEST_ASSERT_NOT_NULL(temp);
    TEST_ASSERT_NOT_NULL(result);

    f32 expected[] = {9, 12, 15, 18};
    for (i32 i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(c);
    tensor_destroy(temp);
    tensor_destroy(result);
}

i32 main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_tensor_op_add_same_shape_no_broadcast);
    RUN_TEST(test_tensor_op_add_with_broadcasting);
    RUN_TEST(test_tensor_op_sub_same_shape);
    RUN_TEST(test_tensor_op_mul_with_broadcasting);
    RUN_TEST(test_tensor_op_div_same_shape);
    RUN_TEST(test_tensor_op_div_by_zero);
    RUN_TEST(test_tensor_op_incompatible_shapes_no_broadcast);
    RUN_TEST(test_tensor_op_scalar_broadcasting);
    RUN_TEST(test_tensor_op_generic_function);
    RUN_TEST(test_tensor_op_gradient_computation);
    RUN_TEST(test_tensor_op_null_inputs);
    RUN_TEST(test_tensor_op_complex_broadcasting);
    RUN_TEST(test_tensor_op_edge_cases);
    RUN_TEST(test_tensor_op_large_tensors);
    RUN_TEST(test_tensor_op_gradient_backward_functions);
    RUN_TEST(test_tensor_op_subtract_gradients);
    RUN_TEST(test_tensor_op_division_gradients);
    RUN_TEST(test_tensor_op_broadcasting_gradients);
    RUN_TEST(test_tensor_op_memory_management);
    RUN_TEST(test_tensor_op_chained_operations);

    return UNITY_END();
}