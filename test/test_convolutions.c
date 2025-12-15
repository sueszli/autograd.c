#include "convolutions.h"
#include "layers.h"
#include "tensor.h"
#include "unity.h"
#include <math.h>
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

void test_conv2d_simple_forward(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    for (int i = 0; i < 9; i++)
        input->data[i] = (float)i;

    Layer *l = layer_conv2d_create(1, 1, 2, 1, 0, false);
    Tensor **params;
    size_t count;
    l->parameters(l, &params, &count);
    for (int i = 0; i < 4; i++)
        params[0]->data[i] = 1.0f;
    free(params);

    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_UINT64(1, out->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(1, out->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(2, out->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(2, out->shape[3]);

    TEST_ASSERT_EQUAL_FLOAT(8.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(12.0f, out->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(20.0f, out->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(24.0f, out->data[3]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_conv2d_padding(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[4] = 1.0f;

    Layer *l = layer_conv2d_create(1, 1, 3, 1, 1, false);
    Tensor **params;
    size_t count;
    l->parameters(l, &params, &count);
    for (int i = 0; i < 9; i++)
        params[0]->data[i] = 1.0f;
    free(params);

    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_UINT64(3, out->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(3, out->shape[3]);

    for (int i = 0; i < 9; i++) {
        TEST_ASSERT_EQUAL_FLOAT(1.0f, out->data[i]);
    }

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_conv2d_stride(void) {
    uint64_t in_shape[] = {1, 1, 4, 4};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    for (int i = 0; i < 16; i++)
        input->data[i] = 1.0f;

    Layer *l = layer_conv2d_create(1, 1, 2, 2, 0, false);
    Tensor **params;
    size_t count;
    l->parameters(l, &params, &count);
    for (int i = 0; i < 4; i++)
        params[0]->data[i] = 1.0f;
    free(params);

    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_UINT64(2, out->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(2, out->shape[3]);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_EQUAL_FLOAT(4.0f, out->data[i]);
    }

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_conv2d_backward(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    for (int i = 0; i < 9; ++i)
        input->data[i] = (float)i;

    uint64_t w_shape[] = {1, 1, 2, 2};
    Tensor *weight = tensor_zeros(w_shape, 4, false);
    for (int i = 0; i < 4; ++i)
        weight->data[i] = 1.0f;

    uint64_t out_shape[] = {1, 1, 2, 2};
    Tensor *grad_out = tensor_zeros(out_shape, 4, false);
    for (int i = 0; i < 4; ++i)
        grad_out->data[i] = 1.0f;

    Tensor *d_in, *d_w, *d_b;
    conv2d_backward(input, weight, NULL, 1, 0, 2, grad_out, &d_in, &d_w, &d_b);

    TEST_ASSERT_EQUAL_FLOAT(8.0f, d_w->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(12.0f, d_w->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(20.0f, d_w->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(24.0f, d_w->data[3]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, d_in->data[4]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, d_in->data[0]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(grad_out);
    tensor_free(d_in);
    tensor_free(d_w);
}

void test_maxpool2d_simple(void) {
    uint64_t in_shape[] = {1, 1, 4, 4};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    for (int i = 0; i < 16; i++)
        input->data[i] = (float)i;

    Layer *l = layer_maxpool2d_create(2, 2, 0);
    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(5.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(7.0f, out->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(13.0f, out->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(15.0f, out->data[3]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_maxpool2d_backward(void) {
    uint64_t in_shape[] = {1, 1, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 1.0f;
    input->data[1] = 2.0f;
    input->data[2] = 3.0f;
    input->data[3] = 4.0f;

    uint64_t out_shape[] = {1, 1, 1, 1};
    uint64_t grad_shape[] = {1, 1, 1, 1};
    Tensor *grad_out = tensor_zeros(grad_shape, 4, false);
    grad_out->data[0] = 10.0f;

    Tensor *grad_in = maxpool2d_backward(input, out_shape, 2, 2, 0, grad_out);

    TEST_ASSERT_EQUAL_FLOAT(0.0f, grad_in->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, grad_in->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, grad_in->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(10.0f, grad_in->data[3]);

    tensor_free(input);
    tensor_free(grad_out);
    tensor_free(grad_in);
}

void test_avgpool2d_simple(void) {
    uint64_t in_shape[] = {1, 1, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    input->data[0] = 0.0f;
    input->data[1] = 4.0f;
    input->data[2] = 0.0f;
    input->data[3] = 4.0f;

    Layer *l = layer_avgpool2d_create(2, 2, 0);
    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_UINT64(1, out->size);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_batchnorm2d_forward(void) {
    uint64_t in_shape[] = {1, 2, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    for (int i = 0; i < 4; ++i)
        input->data[i] = 2.0f;
    input->data[4] = 1.0f;
    input->data[5] = 3.0f;
    input->data[6] = 1.0f;
    input->data[7] = 3.0f;

    Layer *l = layer_batchnorm2d_create(2, 1e-5f, 0.1f);
    Tensor *out = l->forward(l, input, true);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.0f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, -1.0f, out->data[4]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 1.0f, out->data[5]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_simplecnn_smoke(void) {
    uint64_t in_shape[] = {2, 3, 32, 32}; // Batch 2, 32x32 RGB
    Tensor *input = tensor_zeros(in_shape, 4, false);

    Layer *l = simple_cnn_create(10);
    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_UINT64(2, out->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, out->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2048, out->shape[1]);

    TEST_ASSERT_EQUAL_UINT64(2, out->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, out->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2048, out->shape[1]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_conv2d_simple_forward);
    RUN_TEST(test_conv2d_padding);
    RUN_TEST(test_conv2d_stride);
    RUN_TEST(test_conv2d_backward);
    RUN_TEST(test_maxpool2d_simple);
    RUN_TEST(test_maxpool2d_backward);
    RUN_TEST(test_avgpool2d_simple);
    RUN_TEST(test_batchnorm2d_forward);
    RUN_TEST(test_simplecnn_smoke);
    return UNITY_END();
}
