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

void test_tensor_matmul_2x2(void) {
    i32 shape[] = {2, 2};
    f32 data_a[] = {1, 2, 3, 4};
    f32 data_b[] = {5, 6, 7, 8};
    f32 expected[] = {19, 22, 43, 50};

    tensor_t *a = tensor_create(data_a, shape, 2, false);
    tensor_t *b = tensor_create(data_b, shape, 2, false);

    tensor_t *result = tensor_matmul(a, b);

    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(2, result->ndim);
    TEST_ASSERT_EQUAL(2, result->shape[0]);
    TEST_ASSERT_EQUAL(2, result->shape[1]);

    for (i32 i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_matmul_2x3_3x2(void) {
    i32 shape_a[] = {2, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    i32 shape_b[] = {3, 2};
    f32 data_b[] = {7, 8, 9, 10, 11, 12};
    f32 expected[] = {58, 64, 139, 154};

    tensor_t *a = tensor_create(data_a, shape_a, 2, false);
    tensor_t *b = tensor_create(data_b, shape_b, 2, false);

    tensor_t *result = tensor_matmul(a, b);

    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(2, result->ndim);
    TEST_ASSERT_EQUAL(2, result->shape[0]);
    TEST_ASSERT_EQUAL(2, result->shape[1]);

    for (i32 i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_matmul_grad(void) {
    i32 shape_a[] = {2, 3};
    f32 data_a[] = {1, 2, 3, 4, 5, 6};
    i32 shape_b[] = {3, 2};
    f32 data_b[] = {7, 8, 9, 10, 11, 12};

    tensor_t *a = tensor_create(data_a, shape_a, 2, true);
    tensor_t *b = tensor_create(data_b, shape_b, 2, true);

    tensor_t *result = tensor_matmul(a, b);
    result->grad = tensor_create(NULL, result->shape, result->ndim, false);
    for (i32 i = 0; i < 4; i++) {
        result->grad->data[i] = 1.0f;
    }

    result->grad_fn(result);

    TEST_ASSERT_NOT_NULL(a->grad);
    f32 expected_a_grad[] = {15, 19, 23, 15, 19, 23};
    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_a_grad[i], a->grad->data[i]);
    }

    TEST_ASSERT_NOT_NULL(b->grad);
    f32 expected_b_grad[] = {5, 5, 7, 7, 9, 9};
    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_b_grad[i], b->grad->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(result);
}

void test_tensor_relu(void) {
    i32 shape[] = {2, 3};
    f32 data[] = {-1, 2, -3, 4, 0, -6};
    f32 expected[] = {0, 2, 0, 4, 0, 0};

    tensor_t *a = tensor_create(data, shape, 2, false);
    tensor_t *result = tensor_relu(a);

    TEST_ASSERT_NOT_NULL(result);
    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(result);
}

void test_tensor_relu_grad(void) {
    i32 shape[] = {2, 3};
    f32 data[] = {-1, 2, -3, 4, 0, -6};

    tensor_t *a = tensor_create(data, shape, 2, true);
    tensor_t *result = tensor_relu(a);
    result->grad = tensor_create(NULL, result->shape, result->ndim, false);
    for (i32 i = 0; i < 6; i++) {
        result->grad->data[i] = 1.0f;
    }

    result->grad_fn(result);

    TEST_ASSERT_NOT_NULL(a->grad);
    f32 expected_grad[] = {0, 1, 0, 1, 0, 0};
    for (i32 i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_grad[i], a->grad->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(result);
}

void test_tensor_softmax(void) {
    i32 shape[] = {5};
    f32 data[] = {1, 2, 3, 4, 5};
    f32 expected[] = {0.01165623f, 0.03168492f, 0.08612854f, 0.23412166f, 0.63640865f};

    tensor_t *a = tensor_create(data, shape, 1, false);
    tensor_t *result = tensor_softmax(a);

    TEST_ASSERT_NOT_NULL(result);
    for (i32 i = 0; i < 5; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(result);
}

void test_tensor_cross_entropy(void) {
    i32 shape[] = {5};
    f32 data[] = {1, 2, 3, 4, 5};
    i32 target_idx = 2;
    f32 expected = 2.451914f;

    tensor_t *a = tensor_create(data, shape, 1, false);
    tensor_t *result = tensor_cross_entropy(a, target_idx);

    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, expected, result->data[0]);

    tensor_destroy(a);
    tensor_destroy(result);
}

void test_tensor_cross_entropy_grad(void) {
    i32 shape[] = {5};
    f32 data[] = {1, 2, 3, 4, 5};
    i32 target_idx = 2;

    tensor_t *a = tensor_create(data, shape, 1, true);
    tensor_t *loss = tensor_cross_entropy(a, target_idx);
    loss->grad = tensor_create(NULL, loss->shape, loss->ndim, false);
    loss->grad->data[0] = 1.0f;

    loss->grad_fn(loss);

    TEST_ASSERT_NOT_NULL(a->grad);
    f32 expected_grad[] = {0.01165623f, 0.03168492f, -0.91387146f, 0.23412166f, 0.63640865f};
    for (i32 i = 0; i < 5; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected_grad[i], a->grad->data[i]);
    }

    tensor_destroy(a);
    tensor_destroy(loss);
}

void test_tensor_conv2d(void) {
    i32 input_shape[] = {3, 3};
    f32 input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    i32 kernel_shape[] = {2, 2};
    f32 kernel_data[] = {1, 0, 0, 1};
    f32 expected[] = {6, 8, 12, 14};

    tensor_t *input = tensor_create(input_data, input_shape, 2, false);
    tensor_t *kernel = tensor_create(kernel_data, kernel_shape, 2, false);

    tensor_t *result = tensor_conv2d(input, kernel, 1, 0);

    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(2, result->ndim);
    TEST_ASSERT_EQUAL(2, result->shape[0]);
    TEST_ASSERT_EQUAL(2, result->shape[1]);

    for (i32 i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(input);
    tensor_destroy(kernel);
    tensor_destroy(result);
}

void test_tensor_max_pool2d(void) {
    i32 input_shape[] = {4, 4};
    f32 input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    f32 expected[] = {6, 8, 14, 16};

    tensor_t *input = tensor_create(input_data, input_shape, 2, false);

    tensor_t *result = tensor_max_pool2d(input, 2, 2);

    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(2, result->ndim);
    TEST_ASSERT_EQUAL(2, result->shape[0]);
    TEST_ASSERT_EQUAL(2, result->shape[1]);

    for (i32 i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(input);
    tensor_destroy(result);
}

void test_tensor_avg_pool2d(void) {
    i32 input_shape[] = {4, 4};
    f32 input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    f32 expected[] = {3.5, 5.5, 11.5, 13.5};

    tensor_t *input = tensor_create(input_data, input_shape, 2, false);

    tensor_t *result = tensor_avg_pool2d(input, 2, 2);

    TEST_ASSERT_NOT_NULL(result);
    TEST_ASSERT_EQUAL(2, result->ndim);
    TEST_ASSERT_EQUAL(2, result->shape[0]);
    TEST_ASSERT_EQUAL(2, result->shape[1]);

    for (i32 i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, expected[i], result->data[i]);
    }

    tensor_destroy(input);
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
    RUN_TEST(test_tensor_matmul_2x2);
    RUN_TEST(test_tensor_matmul_2x3_3x2);
    RUN_TEST(test_tensor_matmul_grad);
    RUN_TEST(test_tensor_relu);
    RUN_TEST(test_tensor_relu_grad);
    RUN_TEST(test_tensor_softmax);
    RUN_TEST(test_tensor_cross_entropy);
    RUN_TEST(test_tensor_cross_entropy_grad);
    RUN_TEST(test_tensor_conv2d);
    RUN_TEST(test_tensor_max_pool2d);
    RUN_TEST(test_tensor_avg_pool2d);

    return UNITY_END();
}