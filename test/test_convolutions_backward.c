#include "ops/convolutions_backward.h"
#include "tensor.h"
#include "unity.h"
#include <math.h>
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

void test_conv2d_backward_simple(void) {
    uint64_t shape[] = {1, 1, 1, 1};
    Tensor *input = tensor_create(NULL, shape, 4, false);
    input->data[0] = 2.0f;

    Tensor *weight = tensor_create(NULL, shape, 4, false);
    weight->data[0] = 3.0f;

    Tensor *bias = tensor_create(NULL, shape, 1, false);
    bias->data[0] = 0.5f;

    Tensor *grad_output = tensor_create(NULL, shape, 4, false);
    grad_output->data[0] = 1.0f;

    Tensor *d_in = NULL;
    Tensor *d_w = NULL;
    Tensor *d_b = NULL;

    conv2d_backward(input, weight, bias, 1, 0, 1, grad_output, &d_in, &d_w, &d_b);

    TEST_ASSERT_NOT_NULL(d_in);
    TEST_ASSERT_NOT_NULL(d_w);
    TEST_ASSERT_NOT_NULL(d_b);

    TEST_ASSERT_EQUAL_FLOAT(3.0f, d_in->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, d_w->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, d_b->data[0]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(grad_output);
    tensor_free(d_in);
    tensor_free(d_w);
    tensor_free(d_b);
}

void test_conv2d_backward_stride(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    uint64_t w_shape[] = {1, 1, 2, 2};
    Tensor *weight = tensor_zeros(w_shape, 4, false);
    weight->data[0] = 1.0f;
    weight->data[1] = 2.0f;
    weight->data[2] = 3.0f;
    weight->data[3] = 4.0f;

    uint64_t out_shape[] = {1, 1, 1, 1};
    Tensor *grad_output = tensor_zeros(out_shape, 4, false);
    grad_output->data[0] = 10.0f;

    Tensor *d_in = NULL;
    Tensor *d_w = NULL;
    Tensor *d_b = NULL;

    conv2d_backward(input, weight, NULL, 2, 0, 2, grad_output, &d_in, &d_w, &d_b);

    TEST_ASSERT_EQUAL_FLOAT(10.0f, d_in->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(20.0f, d_in->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(30.0f, d_in->data[3]);
    TEST_ASSERT_EQUAL_FLOAT(40.0f, d_in->data[4]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, d_in->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, d_in->data[5]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(grad_output);
    tensor_free(d_in);
    tensor_free(d_w);
}

void test_conv2d_backward_padding(void) {
    uint64_t in_shape[] = {1, 1, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    uint64_t w_shape[] = {1, 1, 3, 3};
    Tensor *weight = tensor_zeros(w_shape, 4, false);
    for (int i = 0; i < 9; ++i)
        weight->data[i] = 1.0f;
    weight->data[4] = 5.0f;

    uint64_t out_shape[] = {1, 1, 1, 1};
    Tensor *grad_output = tensor_zeros(out_shape, 4, false);
    grad_output->data[0] = 2.0f;

    Tensor *d_in = NULL;
    Tensor *d_w = NULL;
    Tensor *d_b = NULL;

    conv2d_backward(input, weight, NULL, 1, 1, 3, grad_output, &d_in, &d_w, &d_b);

    TEST_ASSERT_EQUAL_UINT64(1, d_in->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(1, d_in->shape[3]);
    TEST_ASSERT_EQUAL_FLOAT(10.0f, d_in->data[0]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(grad_output);
    tensor_free(d_in);
    tensor_free(d_w);
}

void test_maxpool2d_backward_simple(void) {
    uint64_t in_shape[] = {1, 1, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 1.0f;
    input->data[1] = 2.0f;
    input->data[2] = 3.0f;
    input->data[3] = 4.0f;

    uint64_t out_shape_arr[] = {1, 1, 1, 1};
    Tensor *grad_output = tensor_zeros(out_shape_arr, 4, false);
    grad_output->data[0] = 10.0f;

    Tensor *d_in = maxpool2d_backward(input, out_shape_arr, 2, 2, 0, grad_output);

    TEST_ASSERT_EQUAL_FLOAT(0.0f, d_in->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, d_in->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, d_in->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(10.0f, d_in->data[3]);

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(d_in);
}

void test_avgpool2d_backward_simple(void) {
    uint64_t in_shape[] = {1, 1, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    uint64_t out_shape_arr[] = {1, 1, 1, 1};
    Tensor *grad_output = tensor_zeros(out_shape_arr, 4, false);
    grad_output->data[0] = 4.0f;

    Tensor *d_in = avgpool2d_backward(input, out_shape_arr, 2, 2, 0, grad_output);

    for (int i = 0; i < 4; ++i) {
        TEST_ASSERT_EQUAL_FLOAT(1.0f, d_in->data[i]);
    }

    tensor_free(input);
    tensor_free(grad_output);
    tensor_free(d_in);
}

void test_batchnorm2d_backward_simple(void) {
    uint64_t shape[] = {1, 1, 2, 2};
    Tensor *input = tensor_zeros(shape, 4, false);
    input->data[0] = 0.0f;
    input->data[1] = 4.0f;
    input->data[2] = 0.0f;
    input->data[3] = 4.0f;

    Tensor *gamma = tensor_zeros(shape, 1, false);
    gamma->shape[0] = 1;
    gamma->ndim = 1;
    gamma->size = 1;
    gamma->data[0] = 1.0f;

    Tensor *mean = tensor_zeros(shape, 1, false);
    mean->shape[0] = 1;
    mean->ndim = 1;
    mean->size = 1;
    mean->data[0] = 2.0f;
    Tensor *var = tensor_zeros(shape, 1, false);
    var->shape[0] = 1;
    var->ndim = 1;
    var->size = 1;
    var->data[0] = 4.0f;

    Tensor *grad_output = tensor_zeros(shape, 4, false);
    for (int i = 0; i < 4; ++i)
        grad_output->data[i] = 1.0f;

    Tensor *d_in = NULL;
    Tensor *d_gamma = NULL;
    Tensor *d_beta = NULL;

    batchnorm2d_backward(input, gamma, mean, var, 0.0f, grad_output, &d_in, &d_gamma, &d_beta);

    TEST_ASSERT_EQUAL_FLOAT(0.0f, d_gamma->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, d_beta->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, d_in->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, d_in->data[1]);

    tensor_free(input);
    tensor_free(gamma);
    tensor_free(mean);
    tensor_free(var);
    tensor_free(grad_output);
    tensor_free(d_in);
    tensor_free(d_gamma);
    tensor_free(d_beta);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_conv2d_backward_simple);
    RUN_TEST(test_conv2d_backward_stride);
    RUN_TEST(test_conv2d_backward_padding);
    RUN_TEST(test_maxpool2d_backward_simple);
    RUN_TEST(test_avgpool2d_backward_simple);
    RUN_TEST(test_batchnorm2d_backward_simple);
    return UNITY_END();
}
