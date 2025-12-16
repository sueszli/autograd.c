#include "autograd.h"
#include "ops/convolutions.h"
#include "ops/reductions.h"
#include "tensor.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_conv2d_backward_simple(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    float32_t in_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    Tensor *input = tensor_create(in_data, in_shape, 4, true);

    uint64_t w_shape[] = {1, 1, 2, 2};
    float32_t w_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
    Tensor *weight = tensor_create(w_data, w_shape, 4, true);

    Tensor *output = tensor_conv2d(input, weight, NULL, 1, 0, 1);
    Tensor *sum1 = tensor_sum(output, 0, false);
    Tensor *sum2 = tensor_sum(sum1, 0, false);
    Tensor *sum3 = tensor_sum(sum2, 0, false);
    Tensor *loss = tensor_sum(sum3, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(input->grad);
    TEST_ASSERT_NOT_NULL(weight->grad);
    TEST_ASSERT_EQUAL_UINT64(4, input->grad->ndim);
    TEST_ASSERT_EQUAL_UINT64(4, weight->grad->ndim);

    tensor_release(input);
    tensor_release(weight);
    tensor_release(output);
    tensor_release(sum1);
    tensor_release(sum2);
    tensor_release(sum3);
    tensor_release(loss);
}

void test_conv2d_backward_with_bias(void) {
    uint64_t in_shape[] = {1, 1, 2, 2};
    float32_t in_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor *input = tensor_create(in_data, in_shape, 4, true);

    uint64_t w_shape[] = {1, 1, 2, 2};
    float32_t w_data[] = {1.0f, 0.0f, 0.0f, 1.0f};
    Tensor *weight = tensor_create(w_data, w_shape, 4, true);

    uint64_t b_shape[] = {1};
    float32_t b_data[] = {1.0f};
    Tensor *bias = tensor_create(b_data, b_shape, 1, true);

    Tensor *output = tensor_conv2d(input, weight, bias, 1, 0, 1);
    Tensor *sum1 = tensor_sum(output, 0, false);
    Tensor *sum2 = tensor_sum(sum1, 0, false);
    Tensor *sum3 = tensor_sum(sum2, 0, false);
    Tensor *loss = tensor_sum(sum3, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(input->grad);
    TEST_ASSERT_NOT_NULL(weight->grad);
    TEST_ASSERT_NOT_NULL(bias->grad);
    TEST_ASSERT_EQUAL_UINT64(1, bias->grad->ndim);

    tensor_release(input);
    tensor_release(weight);
    tensor_release(bias);
    tensor_release(output);
    tensor_release(sum1);
    tensor_release(sum2);
    tensor_release(sum3);
    tensor_release(loss);
}

void test_conv2d_backward_input_only(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    float32_t in_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    Tensor *input = tensor_create(in_data, in_shape, 4, true);

    uint64_t w_shape[] = {1, 1, 2, 2};
    float32_t w_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
    Tensor *weight = tensor_create(w_data, w_shape, 4, false);

    Tensor *output = tensor_conv2d(input, weight, NULL, 1, 0, 1);
    Tensor *sum1 = tensor_sum(output, 0, false);
    Tensor *sum2 = tensor_sum(sum1, 0, false);
    Tensor *sum3 = tensor_sum(sum2, 0, false);
    Tensor *loss = tensor_sum(sum3, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(input->grad);
    TEST_ASSERT_NULL(weight->grad);

    tensor_release(input);
    tensor_release(weight);
    tensor_release(output);
    tensor_release(sum1);
    tensor_release(sum2);
    tensor_release(sum3);
    tensor_release(loss);
}

void test_conv2d_backward_chain(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, true);

    uint64_t w1_shape[] = {1, 1, 2, 2};
    Tensor *weight1 = tensor_zeros(w1_shape, 4, true);

    uint64_t w2_shape[] = {1, 1, 2, 2};
    Tensor *weight2 = tensor_zeros(w2_shape, 4, true);

    Tensor *out1 = tensor_conv2d(input, weight1, NULL, 1, 0, 1);
    Tensor *out2 = tensor_conv2d(out1, weight2, NULL, 1, 0, 1);
    Tensor *sum1 = tensor_sum(out2, 0, false);
    Tensor *sum2 = tensor_sum(sum1, 0, false);
    Tensor *sum3 = tensor_sum(sum2, 0, false);
    Tensor *loss = tensor_sum(sum3, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(input->grad);
    TEST_ASSERT_NOT_NULL(weight1->grad);
    TEST_ASSERT_NOT_NULL(weight2->grad);

    tensor_release(input);
    tensor_release(weight1);
    tensor_release(weight2);
    tensor_release(out1);
    tensor_release(out2);
    tensor_release(sum1);
    tensor_release(sum2);
    tensor_release(sum3);
    tensor_release(loss);
}

void test_maxpool2d_backward_simple(void) {
    uint64_t in_shape[] = {1, 1, 4, 4};
    float32_t in_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    Tensor *input = tensor_create(in_data, in_shape, 4, true);

    Tensor *output = tensor_maxpool2d(input, 2, 2, 0);
    Tensor *sum1 = tensor_sum(output, 0, false);
    Tensor *sum2 = tensor_sum(sum1, 0, false);
    Tensor *sum3 = tensor_sum(sum2, 0, false);
    Tensor *loss = tensor_sum(sum3, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(input->grad);
    TEST_ASSERT_EQUAL_UINT64(4, input->grad->ndim);
    TEST_ASSERT_EQUAL_UINT64(4, input->grad->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(4, input->grad->shape[3]);

    tensor_release(input);
    tensor_release(output);
    tensor_release(sum1);
    tensor_release(sum2);
    tensor_release(sum3);
    tensor_release(loss);
}

void test_maxpool2d_backward_with_padding(void) {
    uint64_t in_shape[] = {1, 1, 2, 2};
    float32_t in_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor *input = tensor_create(in_data, in_shape, 4, true);

    Tensor *output = tensor_maxpool2d(input, 2, 2, 1);
    Tensor *sum1 = tensor_sum(output, 0, false);
    Tensor *sum2 = tensor_sum(sum1, 0, false);
    Tensor *sum3 = tensor_sum(sum2, 0, false);
    Tensor *loss = tensor_sum(sum3, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(input->grad);
    TEST_ASSERT_EQUAL_UINT64(4, input->grad->ndim);

    tensor_release(input);
    tensor_release(output);
    tensor_release(sum1);
    tensor_release(sum2);
    tensor_release(sum3);
    tensor_release(loss);
}

void test_avgpool2d_backward_simple(void) {
    uint64_t in_shape[] = {1, 1, 4, 4};
    float32_t in_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    Tensor *input = tensor_create(in_data, in_shape, 4, true);

    Tensor *output = tensor_avgpool2d(input, 2, 2, 0);
    Tensor *sum1 = tensor_sum(output, 0, false);
    Tensor *sum2 = tensor_sum(sum1, 0, false);
    Tensor *sum3 = tensor_sum(sum2, 0, false);
    Tensor *loss = tensor_sum(sum3, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(input->grad);
    TEST_ASSERT_EQUAL_UINT64(4, input->grad->ndim);
    TEST_ASSERT_EQUAL_UINT64(4, input->grad->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(4, input->grad->shape[3]);

    tensor_release(input);
    tensor_release(output);
    tensor_release(sum1);
    tensor_release(sum2);
    tensor_release(sum3);
    tensor_release(loss);
}

void test_avgpool2d_backward_with_padding(void) {
    uint64_t in_shape[] = {1, 1, 2, 2};
    float32_t in_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor *input = tensor_create(in_data, in_shape, 4, true);

    Tensor *output = tensor_avgpool2d(input, 2, 2, 1);
    Tensor *sum1 = tensor_sum(output, 0, false);
    Tensor *sum2 = tensor_sum(sum1, 0, false);
    Tensor *sum3 = tensor_sum(sum2, 0, false);
    Tensor *loss = tensor_sum(sum3, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(input->grad);
    TEST_ASSERT_EQUAL_UINT64(4, input->grad->ndim);

    tensor_release(input);
    tensor_release(output);
    tensor_release(sum1);
    tensor_release(sum2);
    tensor_release(sum3);
    tensor_release(loss);
}

void test_batchnorm2d_backward_simple(void) {
    uint64_t in_shape[] = {2, 1, 2, 2};
    float32_t in_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Tensor *input = tensor_create(in_data, in_shape, 4, true);

    uint64_t param_shape[] = {1};
    float32_t gamma_data[] = {1.0f};
    float32_t beta_data[] = {0.0f};
    Tensor *gamma = tensor_create(gamma_data, param_shape, 1, true);
    Tensor *beta = tensor_create(beta_data, param_shape, 1, true);

    Tensor *running_mean = tensor_zeros(param_shape, 1, false);
    Tensor *running_var = tensor_zeros(param_shape, 1, false);
    running_var->data[0] = 1.0f;

    Tensor *output = tensor_batchnorm2d(input, gamma, beta, running_mean, running_var, true, 0.1f, 1e-5f);
    Tensor *sum1 = tensor_sum(output, 0, false);
    Tensor *sum2 = tensor_sum(sum1, 0, false);
    Tensor *sum3 = tensor_sum(sum2, 0, false);
    Tensor *loss = tensor_sum(sum3, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(input->grad);
    TEST_ASSERT_NOT_NULL(gamma->grad);
    TEST_ASSERT_NOT_NULL(beta->grad);
    TEST_ASSERT_EQUAL_UINT64(4, input->grad->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, gamma->grad->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, beta->grad->ndim);

    tensor_release(input);
    tensor_release(gamma);
    tensor_release(beta);
    tensor_release(running_mean);
    tensor_release(running_var);
    tensor_release(output);
    tensor_release(sum1);
    tensor_release(sum2);
    tensor_release(sum3);
    tensor_release(loss);
}

void test_batchnorm2d_backward_input_only(void) {
    uint64_t in_shape[] = {2, 1, 2, 2};
    float32_t in_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Tensor *input = tensor_create(in_data, in_shape, 4, true);

    uint64_t param_shape[] = {1};
    float32_t gamma_data[] = {1.0f};
    float32_t beta_data[] = {0.0f};
    Tensor *gamma = tensor_create(gamma_data, param_shape, 1, false);
    Tensor *beta = tensor_create(beta_data, param_shape, 1, false);

    Tensor *running_mean = tensor_zeros(param_shape, 1, false);
    Tensor *running_var = tensor_zeros(param_shape, 1, false);
    running_var->data[0] = 1.0f;

    Tensor *output = tensor_batchnorm2d(input, gamma, beta, running_mean, running_var, true, 0.1f, 1e-5f);
    Tensor *sum1 = tensor_sum(output, 0, false);
    Tensor *sum2 = tensor_sum(sum1, 0, false);
    Tensor *sum3 = tensor_sum(sum2, 0, false);
    Tensor *loss = tensor_sum(sum3, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(input->grad);
    TEST_ASSERT_NULL(gamma->grad);
    TEST_ASSERT_NULL(beta->grad);

    tensor_release(input);
    tensor_release(gamma);
    tensor_release(beta);
    tensor_release(running_mean);
    tensor_release(running_var);
    tensor_release(output);
    tensor_release(sum1);
    tensor_release(sum2);
    tensor_release(sum3);
    tensor_release(loss);
}

void test_conv2d_maxpool_chain(void) {
    uint64_t in_shape[] = {1, 1, 4, 4};
    Tensor *input = tensor_zeros(in_shape, 4, true);

    uint64_t w_shape[] = {1, 1, 2, 2};
    Tensor *weight = tensor_zeros(w_shape, 4, true);

    Tensor *conv_out = tensor_conv2d(input, weight, NULL, 1, 0, 1);
    Tensor *pool_out = tensor_maxpool2d(conv_out, 2, 2, 0);
    Tensor *sum1 = tensor_sum(pool_out, 0, false);
    Tensor *sum2 = tensor_sum(sum1, 0, false);
    Tensor *sum3 = tensor_sum(sum2, 0, false);
    Tensor *loss = tensor_sum(sum3, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(input->grad);
    TEST_ASSERT_NOT_NULL(weight->grad);

    tensor_release(input);
    tensor_release(weight);
    tensor_release(conv_out);
    tensor_release(pool_out);
    tensor_release(sum1);
    tensor_release(sum2);
    tensor_release(sum3);
    tensor_release(loss);
}

void test_conv2d_padding_backward(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, true);

    uint64_t w_shape[] = {1, 1, 3, 3};
    Tensor *weight = tensor_zeros(w_shape, 4, true);

    Tensor *output = tensor_conv2d(input, weight, NULL, 1, 1, 1);
    Tensor *sum1 = tensor_sum(output, 0, false);
    Tensor *sum2 = tensor_sum(sum1, 0, false);
    Tensor *sum3 = tensor_sum(sum2, 0, false);
    Tensor *loss = tensor_sum(sum3, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(input->grad);
    TEST_ASSERT_NOT_NULL(weight->grad);
    TEST_ASSERT_EQUAL_UINT64(3, input->grad->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(3, input->grad->shape[3]);

    tensor_release(input);
    tensor_release(weight);
    tensor_release(output);
    tensor_release(sum1);
    tensor_release(sum2);
    tensor_release(sum3);
    tensor_release(loss);
}

void test_conv2d_stride_backward(void) {
    uint64_t in_shape[] = {1, 1, 5, 5};
    Tensor *input = tensor_zeros(in_shape, 4, true);

    uint64_t w_shape[] = {1, 1, 2, 2};
    Tensor *weight = tensor_zeros(w_shape, 4, true);

    Tensor *output = tensor_conv2d(input, weight, NULL, 2, 0, 1);
    Tensor *sum1 = tensor_sum(output, 0, false);
    Tensor *sum2 = tensor_sum(sum1, 0, false);
    Tensor *sum3 = tensor_sum(sum2, 0, false);
    Tensor *loss = tensor_sum(sum3, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(input->grad);
    TEST_ASSERT_NOT_NULL(weight->grad);

    tensor_release(input);
    tensor_release(weight);
    tensor_release(output);
    tensor_release(sum1);
    tensor_release(sum2);
    tensor_release(sum3);
    tensor_release(loss);
}

void test_conv2d_backward_input_gradient_values(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    float32_t in_data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Tensor *input = tensor_create(in_data, in_shape, 4, true);

    uint64_t w_shape[] = {1, 1, 2, 2};
    float32_t w_data[] = {1, 0, 0, 1};
    Tensor *weight = tensor_create(w_data, w_shape, 4, true);

    Tensor *output = tensor_conv2d(input, weight, NULL, 1, 0, 1);

    Tensor *sum1 = tensor_sum(output, 0, false);
    Tensor *sum2 = tensor_sum(sum1, 0, false);
    Tensor *sum3 = tensor_sum(sum2, 0, false);
    Tensor *loss = tensor_sum(sum3, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(input->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, input->grad->data[4]);

    tensor_release(input);
    tensor_release(weight);
    tensor_release(output);
    tensor_release(sum1);
    tensor_release(sum2);
    tensor_release(sum3);
    tensor_release(loss);
}

void test_conv2d_backward_weight_gradient_values(void) {
    uint64_t in_shape[] = {1, 1, 2, 2};
    float32_t in_data[] = {1, 1, 1, 1};
    Tensor *input = tensor_create(in_data, in_shape, 4, true);

    uint64_t w_shape[] = {1, 1, 2, 2};
    float32_t w_data[] = {0.5f, 0.5f, 0.5f, 0.5f};
    Tensor *weight = tensor_create(w_data, w_shape, 4, true);

    Tensor *output = tensor_conv2d(input, weight, NULL, 1, 0, 1);

    Tensor *sum1 = tensor_sum(output, 0, false);
    Tensor *sum2 = tensor_sum(sum1, 0, false);
    Tensor *sum3 = tensor_sum(sum2, 0, false);
    Tensor *loss = tensor_sum(sum3, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(weight->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, weight->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, weight->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, weight->grad->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, weight->grad->data[3]);

    tensor_release(input);
    tensor_release(weight);
    tensor_release(output);
    tensor_release(sum1);
    tensor_release(sum2);
    tensor_release(sum3);
    tensor_release(loss);
}

void test_batchnorm2d_backward_training_values(void) {
    uint64_t in_shape[] = {1, 1, 2, 2};

    float32_t in_data2[] = {0.0f, 1.0f, 0.0f, 1.0f};
    Tensor *input = tensor_create(in_data2, in_shape, 4, true);

    uint64_t param_shape[] = {1};
    float32_t gamma_data[] = {1.0f};
    float32_t beta_data[] = {0.0f};
    Tensor *gamma = tensor_create(gamma_data, param_shape, 1, true);
    Tensor *beta = tensor_create(beta_data, param_shape, 1, true);
    Tensor *rm = tensor_zeros(param_shape, 1, false);
    Tensor *rv = tensor_zeros(param_shape, 1, false);
    rv->data[0] = 1.0f;

    Tensor *out = tensor_batchnorm2d(input, gamma, beta, rm, rv, true, 0.1f, 1e-5f);

    Tensor *sum1 = tensor_sum(out, 0, false);
    Tensor *sum2 = tensor_sum(sum1, 0, false);
    Tensor *sum3 = tensor_sum(sum2, 0, false);
    Tensor *loss = tensor_sum(sum3, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(input->grad);

    TEST_ASSERT_EQUAL_UINT64(4, input->grad->ndim);

    tensor_release(input);
    tensor_release(gamma);
    tensor_release(beta);
    tensor_release(rm);
    tensor_release(rv);
    tensor_release(out);
    tensor_release(sum1);
    tensor_release(sum2);
    tensor_release(sum3);
    tensor_release(loss);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_conv2d_backward_simple);
    RUN_TEST(test_conv2d_backward_with_bias);
    RUN_TEST(test_conv2d_backward_input_only);
    RUN_TEST(test_conv2d_backward_chain);
    RUN_TEST(test_maxpool2d_backward_simple);
    RUN_TEST(test_maxpool2d_backward_with_padding);
    RUN_TEST(test_avgpool2d_backward_simple);
    RUN_TEST(test_avgpool2d_backward_with_padding);
    RUN_TEST(test_batchnorm2d_backward_simple);
    RUN_TEST(test_batchnorm2d_backward_input_only);
    RUN_TEST(test_conv2d_maxpool_chain);
    RUN_TEST(test_conv2d_padding_backward);
    RUN_TEST(test_conv2d_stride_backward);
    RUN_TEST(test_conv2d_backward_input_gradient_values);
    RUN_TEST(test_conv2d_backward_weight_gradient_values);
    RUN_TEST(test_batchnorm2d_backward_training_values);
    return UNITY_END();
}
