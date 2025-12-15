#include "ops/convolutions.h"
#include "tensor.h"
#include "unity.h"
#include <math.h>
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

static Tensor *create_tensor_from_data(float32_t *data, uint64_t *shape, uint64_t ndim) { return tensor_create(data, shape, ndim, false); }

void test_conv2d_basic_shape(void) {
    uint64_t in_shape[] = {1, 1, 5, 5};
    uint64_t w_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    Tensor *weight = tensor_zeros(w_shape, 4, false);

    Tensor *out = tensor_conv2d(input, weight, NULL, 1, 0, 1);

    TEST_ASSERT_NOT_NULL(out);
    TEST_ASSERT_EQUAL_UINT64(1, out->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(1, out->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(3, out->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(3, out->shape[3]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_conv2d_padding(void) {
    uint64_t in_shape[] = {1, 1, 5, 5};
    uint64_t w_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    Tensor *weight = tensor_zeros(w_shape, 4, false);

    Tensor *out = tensor_conv2d(input, weight, NULL, 1, 1, 1);

    TEST_ASSERT_EQUAL_UINT64(5, out->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(5, out->shape[3]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_conv2d_stride(void) {
    uint64_t in_shape[] = {1, 1, 7, 7};
    uint64_t w_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    Tensor *weight = tensor_zeros(w_shape, 4, false);

    Tensor *out = tensor_conv2d(input, weight, NULL, 2, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(3, out->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(3, out->shape[3]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_conv2d_value_identity(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    float32_t in_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Tensor *input = create_tensor_from_data(in_data, in_shape, 4);

    uint64_t w_shape[] = {1, 1, 1, 1};
    float32_t w_data[] = {1.0f};
    Tensor *weight = create_tensor_from_data(w_data, w_shape, 4);

    Tensor *out = tensor_conv2d(input, weight, NULL, 1, 0, 1);

    for (int i = 0; i < 9; ++i) {
        TEST_ASSERT_EQUAL_FLOAT(in_data[i], out->data[i]);
    }

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_conv2d_value_simple(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    for (int i = 0; i < 9; ++i)
        input->data[i] = 1.0f;

    uint64_t w_shape[] = {1, 1, 2, 2};
    Tensor *weight = tensor_zeros(w_shape, 4, false);
    for (int i = 0; i < 4; ++i)
        weight->data[i] = 1.0f;

    Tensor *out = tensor_conv2d(input, weight, NULL, 1, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(2, out->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(2, out->shape[3]);

    for (int i = 0; i < 4; ++i) {
        TEST_ASSERT_EQUAL_FLOAT(4.0f, out->data[i]);
    }

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_conv2d_bias(void) {
    uint64_t in_shape[] = {1, 1, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    uint64_t w_shape[] = {1, 1, 2, 2};
    Tensor *weight = tensor_zeros(w_shape, 4, false);

    uint64_t b_shape[] = {1};
    Tensor *bias = tensor_zeros(b_shape, 1, false);
    bias->data[0] = 5.0f;

    Tensor *out = tensor_conv2d(input, weight, bias, 1, 0, 1);

    TEST_ASSERT_EQUAL_FLOAT(5.0f, out->data[0]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(out);
}

void test_conv2d_multi_channel_input(void) {
    uint64_t in_shape[] = {1, 2, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    for (uint64_t i = 0; i < input->size; ++i)
        input->data[i] = 1.0f;

    uint64_t w_shape[] = {1, 2, 1, 1};
    Tensor *weight = tensor_zeros(w_shape, 4, false);
    for (uint64_t i = 0; i < weight->size; ++i)
        weight->data[i] = 1.0f;

    Tensor *out = tensor_conv2d(input, weight, NULL, 1, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(1, out->shape[1]);
    for (uint64_t i = 0; i < out->size; ++i) {
        TEST_ASSERT_EQUAL_FLOAT(2.0f, out->data[i]);
    }

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_conv2d_multi_channel_output(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    uint64_t w_shape[] = {2, 1, 1, 1};
    Tensor *weight = tensor_zeros(w_shape, 4, false);

    Tensor *out = tensor_conv2d(input, weight, NULL, 1, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(2, out->shape[1]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_conv2d_batch(void) {
    uint64_t in_shape[] = {2, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 1.0f;
    input->data[9] = 2.0f;

    uint64_t w_shape[] = {1, 1, 1, 1};
    Tensor *weight = tensor_zeros(w_shape, 4, false);
    weight->data[0] = 1.0f;

    Tensor *out = tensor_conv2d(input, weight, NULL, 1, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(2, out->shape[0]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, out->data[9]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_conv2d_dilation(void) {
    uint64_t in_shape[] = {1, 1, 5, 5};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 1.0f;
    input->data[4] = 1.0f;
    input->data[20] = 1.0f;
    input->data[24] = 1.0f;

    uint64_t w_shape[] = {1, 1, 3, 3};
    Tensor *weight = tensor_zeros(w_shape, 4, false);
    weight->data[0] = 1.0f;
    weight->data[2] = 1.0f;
    weight->data[6] = 1.0f;
    weight->data[8] = 1.0f;

    Tensor *out = tensor_conv2d(input, weight, NULL, 1, 0, 2);

    TEST_ASSERT_EQUAL_FLOAT(4.0f, out->data[0]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_maxpool2d_basic(void) {
    uint64_t in_shape[] = {1, 1, 2, 2};
    float32_t data[] = {1, 2, 3, 4};
    Tensor *input = create_tensor_from_data(data, in_shape, 4);

    Tensor *out = tensor_maxpool2d(input, 2, 2, 0);

    TEST_ASSERT_EQUAL_FLOAT(4.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
}

void test_maxpool2d_stride(void) {
    uint64_t in_shape[] = {1, 1, 4, 4};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    for (int i = 0; i < 16; ++i)
        input->data[i] = (float32_t)i;

    Tensor *out = tensor_maxpool2d(input, 2, 2, 0);

    TEST_ASSERT_EQUAL_FLOAT(5.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(7.0f, out->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(13.0f, out->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(15.0f, out->data[3]);

    tensor_free(input);
    tensor_free(out);
}

void test_maxpool2d_padding(void) {
    uint64_t in_shape[] = {1, 1, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 10.0f;

    Tensor *out = tensor_maxpool2d(input, 2, 2, 1);

    TEST_ASSERT_EQUAL_FLOAT(10.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
}

void test_avgpool2d_basic(void) {
    uint64_t in_shape[] = {1, 1, 2, 2};
    float32_t data[] = {1, 2, 3, 4};
    Tensor *input = create_tensor_from_data(data, in_shape, 4);

    Tensor *out = tensor_avgpool2d(input, 2, 2, 0);

    TEST_ASSERT_EQUAL_FLOAT(2.5f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
}

void test_avgpool2d_padding_zeros(void) {
    uint64_t in_shape[] = {1, 1, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 4.0f;

    Tensor *out = tensor_avgpool2d(input, 3, 1, 1);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 4.0f / 9.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
}

void test_batchnorm2d_shape(void) {
    uint64_t in_shape[] = {2, 3, 4, 4};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    uint64_t param_shape[] = {3};
    Tensor *gamma = tensor_zeros(param_shape, 1, false);
    Tensor *beta = tensor_zeros(param_shape, 1, false);
    Tensor *rm = tensor_zeros(param_shape, 1, false);
    Tensor *rv = tensor_zeros(param_shape, 1, false);

    Tensor *out = tensor_batchnorm2d(input, gamma, beta, rm, rv, true, 0.1f, 1e-5f);

    TEST_ASSERT_EQUAL_UINT64(2, out->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, out->shape[1]);

    tensor_free(input);
    tensor_free(gamma);
    tensor_free(beta);
    tensor_free(rm);
    tensor_free(rv);
    tensor_free(out);
}

void test_batchnorm2d_training_mean_var(void) {
    uint64_t in_shape[] = {1, 1, 1, 2};
    float32_t data[] = {1.0f, 3.0f};
    Tensor *input = create_tensor_from_data(data, in_shape, 4);

    uint64_t p_shape[] = {1};
    Tensor *gamma = tensor_zeros(p_shape, 1, false);
    gamma->data[0] = 1.0f;
    Tensor *beta = tensor_zeros(p_shape, 1, false);
    beta->data[0] = 0.0f;
    Tensor *rm = tensor_zeros(p_shape, 1, false);
    Tensor *rv = tensor_zeros(p_shape, 1, false);
    rv->data[0] = 1.0f;

    Tensor *out = tensor_batchnorm2d(input, gamma, beta, rm, rv, true, 0.1f, 0.0f);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, -1.0f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, out->data[1]);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.2f, rm->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, rv->data[0]);

    tensor_free(input);
    tensor_free(gamma);
    tensor_free(beta);
    tensor_free(rm);
    tensor_free(rv);
    tensor_free(out);
}

void test_batchnorm2d_inference(void) {
    uint64_t in_shape[] = {1, 1, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 10.0f;

    uint64_t p_shape[] = {1};
    Tensor *gamma = tensor_zeros(p_shape, 1, false);
    gamma->data[0] = 2.0f;
    Tensor *beta = tensor_zeros(p_shape, 1, false);
    beta->data[0] = 5.0f;
    Tensor *rm = tensor_zeros(p_shape, 1, false);
    rm->data[0] = 0.0f;
    Tensor *rv = tensor_zeros(p_shape, 1, false);
    rv->data[0] = 1.0f;

    Tensor *out = tensor_batchnorm2d(input, gamma, beta, rm, rv, false, 0.1f, 0.0f);

    TEST_ASSERT_EQUAL_FLOAT(25.0f, out->data[0]);

    tensor_free(input);
    tensor_free(gamma);
    tensor_free(beta);
    tensor_free(rm);
    tensor_free(rv);
    tensor_free(out);
}

void test_batchnorm2d_gamma_beta(void) {
    uint64_t in_shape[] = {1, 1, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 10.0f;

    uint64_t p_shape[] = {1};
    Tensor *gamma = tensor_zeros(p_shape, 1, false);
    gamma->data[0] = 0.0f;
    Tensor *beta = tensor_zeros(p_shape, 1, false);
    beta->data[0] = 7.0f;
    Tensor *rm = tensor_zeros(p_shape, 1, false);
    Tensor *rv = tensor_zeros(p_shape, 1, false);

    Tensor *out = tensor_batchnorm2d(input, gamma, beta, rm, rv, true, 0.1f, 1e-5f);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 7.0f, out->data[0]);

    tensor_free(input);
    tensor_free(gamma);
    tensor_free(beta);
    tensor_free(rm);
    tensor_free(rv);
    tensor_free(out);
}

void test_conv2d_large_padding(void) {
    uint64_t in_shape[] = {1, 1, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    uint64_t w_shape[] = {1, 1, 1, 1};
    Tensor *weight = tensor_zeros(w_shape, 4, false);

    Tensor *out = tensor_conv2d(input, weight, NULL, 1, 10, 1);

    TEST_ASSERT_EQUAL_UINT64(21, out->shape[2]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_conv2d_requires_grad_propagation(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, true);

    uint64_t w_shape[] = {1, 1, 1, 1};
    Tensor *weight = tensor_zeros(w_shape, 4, false);

    Tensor *out = tensor_conv2d(input, weight, NULL, 1, 0, 1);

    TEST_ASSERT_TRUE(out->requires_grad);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_conv2d_random_large(void) {
    uint64_t in_shape[] = {2, 4, 10, 10};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    uint64_t w_shape[] = {8, 4, 3, 3};
    Tensor *weight = tensor_zeros(w_shape, 4, false);

    Tensor *out = tensor_conv2d(input, weight, NULL, 1, 1, 1);

    TEST_ASSERT_EQUAL_UINT64(2, out->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(8, out->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(10, out->shape[2]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_maxpool_random_large(void) {
    uint64_t in_shape[] = {2, 4, 32, 32};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    Tensor *out = tensor_maxpool2d(input, 2, 2, 0);

    TEST_ASSERT_EQUAL_UINT64(2, out->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(4, out->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(16, out->shape[2]);

    tensor_free(input);
    tensor_free(out);
}

void test_conv2d_kernel_larger_than_input(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    uint64_t w_shape[] = {1, 1, 5, 5};
    Tensor *weight = tensor_zeros(w_shape, 4, false);

    Tensor *out = tensor_conv2d(input, weight, NULL, 1, 1, 1);

    TEST_ASSERT_EQUAL_UINT64(1, out->shape[2]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_avgpool_irregular_stride(void) {
    uint64_t in_shape[] = {1, 1, 10, 10};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    Tensor *out = tensor_avgpool2d(input, 3, 2, 0);

    TEST_ASSERT_EQUAL_UINT64(4, out->shape[2]);

    tensor_free(input);
    tensor_free(out);
}

void test_maxpool_kernel_1(void) {
    uint64_t in_shape[] = {1, 1, 5, 5};
    float32_t val[25];
    for (int i = 0; i < 25; ++i)
        val[i] = (float32_t)i;

    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 5.0f;

    Tensor *out = tensor_maxpool2d(input, 1, 1, 0);

    TEST_ASSERT_EQUAL_UINT64(5, out->shape[2]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
}

void test_conv2d_1x1_stride2(void) {
    uint64_t in_shape[] = {1, 1, 4, 4};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 1.0f;
    input->data[1] = 2.0f;

    uint64_t w_shape[] = {1, 1, 1, 1};
    Tensor *weight = tensor_zeros(w_shape, 4, false);
    weight->data[0] = 1.0f;

    Tensor *out = tensor_conv2d(input, weight, NULL, 2, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(2, out->shape[2]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, out->data[1]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_padding_function_logic(void) {
    uint64_t in_shape[] = {1, 1, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 1.0f;

    Tensor *out = tensor_avgpool2d(input, 3, 1, 1);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f / 9.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
}

void test_conv2d_asymmetric_input(void) {
    uint64_t in_shape[] = {1, 1, 5, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    uint64_t w_shape[] = {1, 1, 3, 3};
    Tensor *weight = tensor_zeros(w_shape, 4, false);

    Tensor *out = tensor_conv2d(input, weight, NULL, 1, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(3, out->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(1, out->shape[3]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_conv2d_dilation_gap(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[1] = 100.0f;

    uint64_t w_shape[] = {1, 1, 2, 2};
    Tensor *weight = tensor_zeros(w_shape, 4, false);
    weight->data[0] = 1.0f;
    weight->data[1] = 1.0f;
    weight->data[2] = 1.0f;
    weight->data[3] = 1.0f;

    Tensor *out = tensor_conv2d(input, weight, NULL, 1, 0, 2);

    TEST_ASSERT_EQUAL_FLOAT(0.0f, out->data[0]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_maxpool_neg_inf(void) {
    uint64_t in_shape[] = {1, 1, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    for (int i = 0; i < 4; ++i)
        input->data[i] = -100.0f;

    Tensor *out = tensor_maxpool2d(input, 2, 2, 0);
    TEST_ASSERT_EQUAL_FLOAT(-100.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
}

void test_conv2d_deep_channels(void) {
    uint64_t in_shape[] = {1, 10, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    uint64_t w_shape[] = {20, 10, 1, 1};
    Tensor *weight = tensor_zeros(w_shape, 4, false);

    Tensor *out = tensor_conv2d(input, weight, NULL, 1, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(20, out->shape[1]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(out);
}

void test_batchnorm_eps(void) {
    uint64_t in_shape[] = {1, 1, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    uint64_t p_shape[] = {1};
    Tensor *gamma = tensor_zeros(p_shape, 1, false);
    gamma->data[0] = 1.0f;
    Tensor *beta = tensor_zeros(p_shape, 1, false);
    Tensor *rm = tensor_zeros(p_shape, 1, false);
    Tensor *rv = tensor_zeros(p_shape, 1, false);

    Tensor *out = tensor_batchnorm2d(input, gamma, beta, rm, rv, true, 0.1f, 1.0f);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.0f, out->data[0]);

    tensor_free(input);
    tensor_free(gamma);
    tensor_free(beta);
    tensor_free(rm);
    tensor_free(rv);
    tensor_free(out);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_conv2d_basic_shape);
    RUN_TEST(test_conv2d_padding);
    RUN_TEST(test_conv2d_stride);
    RUN_TEST(test_conv2d_value_identity);
    RUN_TEST(test_conv2d_value_simple);
    RUN_TEST(test_conv2d_bias);
    RUN_TEST(test_conv2d_multi_channel_input);
    RUN_TEST(test_conv2d_multi_channel_output);
    RUN_TEST(test_conv2d_batch);
    RUN_TEST(test_conv2d_dilation);
    RUN_TEST(test_maxpool2d_basic);
    RUN_TEST(test_maxpool2d_stride);
    RUN_TEST(test_maxpool2d_padding);
    RUN_TEST(test_avgpool2d_basic);
    RUN_TEST(test_avgpool2d_padding_zeros);
    RUN_TEST(test_batchnorm2d_shape);
    RUN_TEST(test_batchnorm2d_training_mean_var);
    RUN_TEST(test_batchnorm2d_inference);
    RUN_TEST(test_batchnorm2d_gamma_beta);
    RUN_TEST(test_conv2d_large_padding);
    RUN_TEST(test_conv2d_requires_grad_propagation);
    RUN_TEST(test_conv2d_random_large);
    RUN_TEST(test_maxpool_random_large);
    RUN_TEST(test_conv2d_kernel_larger_than_input);
    RUN_TEST(test_avgpool_irregular_stride);
    RUN_TEST(test_maxpool_kernel_1);
    RUN_TEST(test_conv2d_1x1_stride2);
    RUN_TEST(test_padding_function_logic);
    RUN_TEST(test_conv2d_asymmetric_input);
    RUN_TEST(test_conv2d_dilation_gap);
    RUN_TEST(test_maxpool_neg_inf);
    RUN_TEST(test_conv2d_deep_channels);
    RUN_TEST(test_batchnorm_eps);
    return UNITY_END();
}
