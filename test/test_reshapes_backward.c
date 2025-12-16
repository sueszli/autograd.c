#include "ops/reshapes.h"
#include "ops/reshapes_backward.h"
#include "tensor.h"
#include "unity.h"
#include <math.h>
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

static void test_reshape_backward_1d_to_1d(void) {
    uint64_t shape_in[] = {5};
    Tensor *input = tensor_create(NULL, shape_in, 1, true);
    Tensor *grad_output = tensor_create(NULL, shape_in, 1, false);
    Tensor *grad_input = tensor_reshape_backward(grad_output, input);

    TEST_ASSERT_EQUAL_UINT64(1, grad_input->ndim);
    TEST_ASSERT_EQUAL_UINT64(5, grad_input->shape[0]);

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_reshape_backward_2d_to_1d(void) {
    uint64_t shape_in[] = {2, 3};
    Tensor *input = tensor_create(NULL, shape_in, 2, true);
    uint64_t shape_out[] = {6};
    Tensor *grad_output = tensor_create(NULL, shape_out, 1, false);
    Tensor *grad_input = tensor_reshape_backward(grad_output, input);

    TEST_ASSERT_EQUAL_UINT64(2, grad_input->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, grad_input->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, grad_input->shape[1]);

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_reshape_backward_1d_to_2d(void) {
    uint64_t shape_in[] = {6};
    Tensor *input = tensor_create(NULL, shape_in, 1, true);
    uint64_t shape_out[] = {2, 3};
    Tensor *grad_output = tensor_create(NULL, shape_out, 2, false);
    Tensor *grad_input = tensor_reshape_backward(grad_output, input);

    TEST_ASSERT_EQUAL_UINT64(1, grad_input->ndim);
    TEST_ASSERT_EQUAL_UINT64(6, grad_input->shape[0]);

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_reshape_backward_3d_to_1d(void) {
    uint64_t shape_in[] = {2, 2, 2};
    Tensor *input = tensor_create(NULL, shape_in, 3, true);
    uint64_t shape_out[] = {8};
    Tensor *grad_output = tensor_create(NULL, shape_out, 1, false);
    Tensor *grad_input = tensor_reshape_backward(grad_output, input);

    TEST_ASSERT_EQUAL_UINT64(3, grad_input->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, grad_input->shape[0]);

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_reshape_backward_1d_to_3d(void) {
    uint64_t shape_in[] = {8};
    Tensor *input = tensor_create(NULL, shape_in, 1, true);
    uint64_t shape_out[] = {2, 2, 2};
    Tensor *grad_output = tensor_create(NULL, shape_out, 3, false);
    Tensor *grad_input = tensor_reshape_backward(grad_output, input);

    TEST_ASSERT_EQUAL_UINT64(1, grad_input->ndim);
    TEST_ASSERT_EQUAL_UINT64(8, grad_input->shape[0]);

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_reshape_backward_identity(void) {
    uint64_t shape_in[] = {3, 3};
    Tensor *input = tensor_create(NULL, shape_in, 2, true);
    Tensor *grad_output = tensor_create(NULL, shape_in, 2, false);
    Tensor *grad_input = tensor_reshape_backward(grad_output, input);

    TEST_ASSERT_EQUAL_UINT64(2, grad_input->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, grad_input->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, grad_input->shape[1]);

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_reshape_backward_flatten(void) {
    uint64_t shape_in[] = {5, 5};
    Tensor *input = tensor_create(NULL, shape_in, 2, true);
    uint64_t shape_out[] = {25};
    Tensor *grad_output = tensor_create(NULL, shape_out, 1, false);
    Tensor *grad_input = tensor_reshape_backward(grad_output, input);

    TEST_ASSERT_EQUAL_UINT64(2, grad_input->ndim);
    TEST_ASSERT_EQUAL_UINT64(5, grad_input->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(5, grad_input->shape[1]);

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_reshape_backward_scalar(void) {
    uint64_t shape_in[] = {1};
    Tensor *input = tensor_create(NULL, shape_in, 1, true);
    uint64_t shape_out[] = {1, 1};
    Tensor *grad_output = tensor_create(NULL, shape_out, 2, false);
    Tensor *grad_input = tensor_reshape_backward(grad_output, input);

    TEST_ASSERT_EQUAL_UINT64(1, grad_input->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, grad_input->shape[0]);

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_reshape_backward_scalar_collapse(void) {
    uint64_t shape_in[] = {1, 1};
    Tensor *input = tensor_create(NULL, shape_in, 2, true);
    uint64_t shape_out[] = {1};
    Tensor *grad_output = tensor_create(NULL, shape_out, 1, false);
    Tensor *grad_input = tensor_reshape_backward(grad_output, input);

    TEST_ASSERT_EQUAL_UINT64(2, grad_input->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, grad_input->shape[0]);

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_reshape_backward_inferred(void) {
    uint64_t shape_in[] = {10};
    Tensor *input = tensor_create(NULL, shape_in, 1, true);
    uint64_t shape_out[] = {2, 5};
    Tensor *grad_output = tensor_create(NULL, shape_out, 2, false);
    Tensor *grad_input = tensor_reshape_backward(grad_output, input);

    TEST_ASSERT_EQUAL_UINT64(1, grad_input->ndim);
    TEST_ASSERT_EQUAL_UINT64(10, grad_input->shape[0]);

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_reshape_backward_large_dims(void) {
    uint64_t shape_in[] = {1000};
    Tensor *input = tensor_create(NULL, shape_in, 1, true);
    uint64_t shape_out[] = {10, 100};
    Tensor *grad_output = tensor_create(NULL, shape_out, 2, false);
    Tensor *grad_input = tensor_reshape_backward(grad_output, input);

    TEST_ASSERT_EQUAL_UINT64(1, grad_input->ndim);
    TEST_ASSERT_EQUAL_UINT64(1000, grad_input->shape[0]);

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_reshape_backward_values(void) {
    uint64_t shape_in[] = {4};
    Tensor *input = tensor_create(NULL, shape_in, 1, true);
    uint64_t shape_out[] = {2, 2};
    Tensor *grad_output = tensor_create(NULL, shape_out, 2, false);
    grad_output->data[0] = 1.0f;
    grad_output->data[1] = 2.0f;
    grad_output->data[2] = 3.0f;
    grad_output->data[3] = 4.0f;

    Tensor *grad_input = tensor_reshape_backward(grad_output, input);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, grad_input->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 2.0f, grad_input->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.0f, grad_input->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 4.0f, grad_input->data[3]);

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_reshape_backward_negative_values(void) {
    uint64_t shape_in[] = {2};
    Tensor *input = tensor_create(NULL, shape_in, 1, true);
    uint64_t shape_out[] = {2};
    Tensor *grad_output = tensor_create(NULL, shape_out, 1, false);
    grad_output->data[0] = -1.5f;
    grad_output->data[1] = -2.5f;

    Tensor *grad_input = tensor_reshape_backward(grad_output, input);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, -1.5f, grad_input->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -2.5f, grad_input->data[1]);

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_reshape_backward_zero_tensor(void) {
    uint64_t shape_in[] = {1};
    Tensor *input = tensor_create(NULL, shape_in, 1, true);
    uint64_t shape_out[] = {1};
    Tensor *grad_output = tensor_create(NULL, shape_out, 1, false);
    grad_output->data[0] = 0.0f;

    Tensor *grad_input = tensor_reshape_backward(grad_output, input);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.0f, grad_input->data[0]);

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_reshape_backward_mixed_dims(void) {
    uint64_t shape_in[] = {2, 3, 2};
    Tensor *input = tensor_create(NULL, shape_in, 3, true);
    uint64_t shape_out[] = {4, 3};
    Tensor *grad_output = tensor_create(NULL, shape_out, 2, false);

    Tensor *grad_input = tensor_reshape_backward(grad_output, input);
    TEST_ASSERT_EQUAL_UINT64(3, grad_input->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, grad_input->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, grad_input->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(2, grad_input->shape[2]);

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_transpose_backward_2d_0_1(void) {
    uint64_t shape_grad[] = {3, 2};
    Tensor *grad_output = tensor_create(NULL, shape_grad, 2, false);
    Tensor *grad_input = tensor_transpose_backward(grad_output, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(2, grad_input->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, grad_input->shape[1]);

    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_transpose_backward_square(void) {
    uint64_t shape_grad[] = {3, 3};
    Tensor *grad_output = tensor_create(NULL, shape_grad, 2, false);
    Tensor *grad_input = tensor_transpose_backward(grad_output, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(3, grad_input->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, grad_input->shape[1]);

    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_transpose_backward_identity_dims(void) {
    uint64_t shape_grad[] = {2, 3};
    Tensor *grad_output = tensor_create(NULL, shape_grad, 2, false);
    Tensor *grad_input = tensor_transpose_backward(grad_output, 0, 0);

    TEST_ASSERT_EQUAL_UINT64(2, grad_input->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, grad_input->shape[1]);

    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_transpose_backward_3d_0_1(void) {
    uint64_t shape_grad[] = {3, 2, 4};
    Tensor *grad_output = tensor_create(NULL, shape_grad, 3, false);
    Tensor *grad_input = tensor_transpose_backward(grad_output, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(2, grad_input->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, grad_input->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(4, grad_input->shape[2]);

    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_transpose_backward_3d_0_2(void) {
    uint64_t shape_grad[] = {4, 3, 2};
    Tensor *grad_output = tensor_create(NULL, shape_grad, 3, false);
    Tensor *grad_input = tensor_transpose_backward(grad_output, 0, 2);

    TEST_ASSERT_EQUAL_UINT64(2, grad_input->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, grad_input->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(4, grad_input->shape[2]);

    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_transpose_backward_3d_1_2(void) {
    uint64_t shape_grad[] = {2, 4, 3};
    Tensor *grad_output = tensor_create(NULL, shape_grad, 3, false);
    Tensor *grad_input = tensor_transpose_backward(grad_output, 1, 2);

    TEST_ASSERT_EQUAL_UINT64(2, grad_input->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, grad_input->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(4, grad_input->shape[2]);

    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_transpose_backward_4d(void) {
    uint64_t shape_grad[] = {5, 3, 4, 2};
    Tensor *grad_output = tensor_create(NULL, shape_grad, 4, false);
    Tensor *grad_input = tensor_transpose_backward(grad_output, 0, 3);

    TEST_ASSERT_EQUAL_UINT64(2, grad_input->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, grad_input->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(4, grad_input->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(5, grad_input->shape[3]);

    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_transpose_backward_double(void) {
    uint64_t shape[] = {2, 3};
    Tensor *grad_output = tensor_create(NULL, shape, 2, false);
    Tensor *step1 = tensor_transpose_backward(grad_output, 0, 1);
    Tensor *step2 = tensor_transpose_backward(step1, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(2, step2->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, step2->shape[1]);

    tensor_free(grad_output);
    tensor_free(step1);
    tensor_free(step2);
}

static void test_transpose_backward_values_check(void) {
    uint64_t shape_grad[] = {3, 2};
    Tensor *grad_output = tensor_create(NULL, shape_grad, 2, false);
    for (int i = 0; i < 6; i++)
        grad_output->data[i] = (float)i;

    Tensor *grad_input = tensor_transpose_backward(grad_output, 0, 1);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.0f, grad_input->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 2.0f, grad_input->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 4.0f, grad_input->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, grad_input->data[3]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.0f, grad_input->data[4]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 5.0f, grad_input->data[5]);

    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_transpose_backward_row_vector(void) {
    uint64_t shape_grad[] = {5, 1};
    Tensor *grad_output = tensor_create(NULL, shape_grad, 2, false);
    Tensor *grad_input = tensor_transpose_backward(grad_output, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(1, grad_input->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(5, grad_input->shape[1]);

    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_transpose_backward_col_vector(void) {
    uint64_t shape_grad[] = {1, 5};
    Tensor *grad_output = tensor_create(NULL, shape_grad, 2, false);
    Tensor *grad_input = tensor_transpose_backward(grad_output, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(5, grad_input->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(1, grad_input->shape[1]);

    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_transpose_backward_single_element(void) {
    uint64_t shape_grad[] = {1, 1};
    Tensor *grad_output = tensor_create(NULL, shape_grad, 2, false);
    grad_output->data[0] = 3.14f;
    Tensor *grad_input = tensor_transpose_backward(grad_output, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(1, grad_input->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(1, grad_input->shape[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.14f, grad_input->data[0]);

    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_transpose_backward_large_matrix(void) {
    uint64_t shape_grad[] = {100, 100};
    Tensor *grad_output = tensor_create(NULL, shape_grad, 2, false);
    Tensor *grad_input = tensor_transpose_backward(grad_output, 0, 1);
    TEST_ASSERT_EQUAL_UINT64(100, grad_input->shape[0]);
    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_transpose_backward_high_dim_values(void) {
    uint64_t shape_grad[] = {2, 1, 2};
    Tensor *grad_output = tensor_create(NULL, shape_grad, 3, false);
    for (int i = 0; i < 4; i++)
        grad_output->data[i] = (float)i;

    Tensor *grad_input = tensor_transpose_backward(grad_output, 0, 2);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.0f, grad_input->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 2.0f, grad_input->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, grad_input->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.0f, grad_input->data[3]);

    tensor_free(grad_output);
    tensor_free(grad_input);
}

static void test_transpose_backward_strides_check(void) {
    uint64_t shape_grad[] = {3, 2};
    Tensor *grad_output = tensor_create(NULL, shape_grad, 2, false);
    Tensor *grad_input = tensor_transpose_backward(grad_output, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(3, grad_input->strides[0]);
    TEST_ASSERT_EQUAL_UINT64(1, grad_input->strides[1]);

    tensor_free(grad_output);
    tensor_free(grad_input);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_reshape_backward_1d_to_1d);
    RUN_TEST(test_reshape_backward_2d_to_1d);
    RUN_TEST(test_reshape_backward_1d_to_2d);
    RUN_TEST(test_reshape_backward_3d_to_1d);
    RUN_TEST(test_reshape_backward_1d_to_3d);
    RUN_TEST(test_reshape_backward_identity);
    RUN_TEST(test_reshape_backward_flatten);
    RUN_TEST(test_reshape_backward_scalar);
    RUN_TEST(test_reshape_backward_scalar_collapse);
    RUN_TEST(test_reshape_backward_inferred);
    RUN_TEST(test_reshape_backward_large_dims);
    RUN_TEST(test_reshape_backward_values);
    RUN_TEST(test_reshape_backward_negative_values);
    RUN_TEST(test_reshape_backward_zero_tensor);
    RUN_TEST(test_reshape_backward_mixed_dims);
    RUN_TEST(test_transpose_backward_2d_0_1);
    RUN_TEST(test_transpose_backward_square);
    RUN_TEST(test_transpose_backward_identity_dims);
    RUN_TEST(test_transpose_backward_3d_0_1);
    RUN_TEST(test_transpose_backward_3d_0_2);
    RUN_TEST(test_transpose_backward_3d_1_2);
    RUN_TEST(test_transpose_backward_4d);
    RUN_TEST(test_transpose_backward_double);
    RUN_TEST(test_transpose_backward_values_check);
    RUN_TEST(test_transpose_backward_row_vector);
    RUN_TEST(test_transpose_backward_col_vector);
    RUN_TEST(test_transpose_backward_single_element);
    RUN_TEST(test_transpose_backward_large_matrix);
    RUN_TEST(test_transpose_backward_high_dim_values);
    RUN_TEST(test_transpose_backward_strides_check);
    return UNITY_END();
}
