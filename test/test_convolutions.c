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
    uint64_t in_shape[] = {2, 3, 32, 32};
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

void test_conv2d_3x3_stride1_pad1(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[4] = 1.0f;

    Layer *l = layer_conv2d_create(1, 1, 3, 1, 1, false);
    Tensor **params;
    size_t count;
    l->parameters(l, &params, &count);

    for (int i = 0; i < 9; ++i)
        params[0]->data[i] = 1.0f;
    free(params);

    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_UINT64(3, out->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(3, out->shape[3]);

    for (int i = 0; i < 9; ++i) {
        TEST_ASSERT_EQUAL_FLOAT(1.0f, out->data[i]);
    }

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_conv2d_kernel_larger_than_input_with_pad(void) {
    uint64_t in_shape[] = {1, 1, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    for (int i = 0; i < 4; ++i)
        input->data[i] = 1.0f;

    Layer *l = layer_conv2d_create(1, 1, 3, 1, 1, false);
    Tensor **params;
    size_t count;
    l->parameters(l, &params, &count);
    for (int i = 0; i < 9; ++i)
        params[0]->data[i] = 1.0f;
    free(params);

    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_UINT64(2, out->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(2, out->shape[3]);

    TEST_ASSERT_EQUAL_FLOAT(4.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, out->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, out->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, out->data[3]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_conv2d_batch_size_multi(void) {
    uint64_t in_shape[] = {2, 1, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    for (int i = 0; i < 4; ++i)
        input->data[i] = 1.0f;
    for (int i = 4; i < 8; ++i)
        input->data[i] = 2.0f;

    Layer *l = layer_conv2d_create(1, 1, 2, 1, 0, false);
    Tensor **params;
    size_t count;
    l->parameters(l, &params, &count);
    for (int i = 0; i < 4; ++i)
        params[0]->data[i] = 0.5f;
    free(params);

    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_UINT64(2, out->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(1, out->shape[2]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, out->data[1]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_conv2d_multi_channel(void) {
    uint64_t in_shape[] = {1, 2, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    for (int i = 0; i < 4; ++i)
        input->data[i] = 1.0f;
    for (int i = 4; i < 8; ++i)
        input->data[i] = 2.0f;

    Layer *l = layer_conv2d_create(2, 2, 2, 1, 0, false);
    Tensor **params;
    size_t count;
    l->parameters(l, &params, &count);

    for (int i = 0; i < 16; ++i)
        params[0]->data[i] = 1.0f;
    free(params);

    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(12.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(12.0f, out->data[1]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_conv2d_bias_value(void) {
    uint64_t in_shape[] = {1, 1, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 0.0f;

    Layer *l = layer_conv2d_create(1, 1, 1, 1, 0, true);
    Tensor **params;
    size_t count;
    l->parameters(l, &params, &count);
    params[1]->data[0] = 100.0f;
    free(params);

    Tensor *out = l->forward(l, input, false);
    TEST_ASSERT_EQUAL_FLOAT(100.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_conv2d_negative_input(void) {
    uint64_t in_shape[] = {1, 1, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = -5.0f;

    Layer *l = layer_conv2d_create(1, 1, 1, 1, 0, false);
    Tensor **params;
    size_t count;
    l->parameters(l, &params, &count);
    params[0]->data[0] = -2.0f;
    free(params);

    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(10.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_conv2d_large_padding(void) {
    uint64_t in_shape[] = {1, 1, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 1.0f;

    Layer *l = layer_conv2d_create(1, 1, 1, 1, 2, false);
    Tensor **params;
    size_t count;
    l->parameters(l, &params, &count);
    params[0]->data[0] = 1.0f;
    free(params);

    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_UINT64(5, out->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(5, out->shape[3]);

    TEST_ASSERT_EQUAL_FLOAT(1.0f, out->data[12]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_conv2d_rect_input(void) {
    uint64_t in_shape[] = {1, 1, 4, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    Layer *l = layer_conv2d_create(1, 1, 2, 1, 0, false);

    Tensor *out = l->forward(l, input, false);
    TEST_ASSERT_EQUAL_UINT64(3, out->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(2, out->shape[3]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_conv2d_1x1_kernel(void) {
    uint64_t in_shape[] = {1, 3, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    for (int i = 0; i < 3 * 2 * 2; ++i)
        input->data[i] = (float)i;

    Layer *l = layer_conv2d_create(3, 1, 1, 1, 0, false);
    Tensor **params;
    size_t count;
    l->parameters(l, &params, &count);
    params[0]->data[0] = 1.0f;
    params[0]->data[1] = 0.0f;
    params[0]->data[2] = 0.0f;
    free(params);

    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(0.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, out->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, out->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, out->data[3]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_conv2d_deep_channels(void) {
    uint64_t in_shape[] = {1, 16, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    Layer *l = layer_conv2d_create(16, 1, 1, 1, 0, false);
    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_UINT64(1, out->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(1, out->size / 4);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_conv2d_backward_bias_accum(void) {
    uint64_t in_shape[] = {2, 1, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    Layer *l = layer_conv2d_create(1, 1, 2, 1, 0, true);
    uint64_t w_shape[] = {1, 1, 2, 2};
    Tensor *weight = tensor_zeros(w_shape, 4, false);

    uint64_t b_shape[] = {1};
    Tensor *bias = tensor_zeros(b_shape, 1, false);

    uint64_t gout_shape[] = {2, 1, 1, 1};
    Tensor *grad_out = tensor_zeros(gout_shape, 4, false);
    grad_out->data[0] = 5.0f;
    grad_out->data[1] = 10.0f;

    Tensor *d_in, *d_w, *d_b;
    conv2d_backward(input, weight, bias, 1, 0, 2, grad_out, &d_in, &d_w, &d_b);

    TEST_ASSERT_EQUAL_FLOAT(15.0f, d_b->data[0]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(bias);
    tensor_free(grad_out);
    tensor_free(d_in);
    tensor_free(d_w);
    tensor_free(d_b);
    l->free(l);
}

void test_conv2d_backward_grad_input_shape(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    uint64_t w_shape[] = {1, 1, 2, 2};
    Tensor *weight = tensor_zeros(w_shape, 4, false);
    uint64_t gout_shape[] = {1, 1, 2, 2};
    Tensor *grad_out = tensor_zeros(gout_shape, 4, false);

    Tensor *d_in, *d_w, *d_b;
    conv2d_backward(input, weight, NULL, 1, 0, 2, grad_out, &d_in, &d_w, &d_b);

    TEST_ASSERT_EQUAL_UINT64(input->size, d_in->size);
    TEST_ASSERT_EQUAL_UINT64(3, d_in->shape[2]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(grad_out);
    tensor_free(d_in);
    tensor_free(d_w);
}

void test_conv2d_backward_grad_weight_shape(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    uint64_t w_shape[] = {2, 1, 2, 2};
    Tensor *weight = tensor_zeros(w_shape, 4, false);
    uint64_t gout_shape[] = {1, 2, 2, 2};
    Tensor *grad_out = tensor_zeros(gout_shape, 4, false);

    Tensor *d_in, *d_w, *d_b;
    conv2d_backward(input, weight, NULL, 1, 0, 2, grad_out, &d_in, &d_w, &d_b);

    TEST_ASSERT_EQUAL_UINT64(weight->size, d_w->size);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(grad_out);
    tensor_free(d_in);
    tensor_free(d_w);
}

void test_conv2d_backward_strided(void) {
    uint64_t in_shape[] = {1, 1, 4, 4};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 1.0f;

    uint64_t w_shape[] = {1, 1, 2, 2};
    Tensor *weight = tensor_zeros(w_shape, 4, false);
    for (int i = 0; i < 4; ++i)
        weight->data[i] = 1.0f;

    uint64_t gout_shape[] = {1, 1, 2, 2};
    Tensor *grad_out = tensor_zeros(gout_shape, 4, false);
    for (int i = 0; i < 4; ++i)
        grad_out->data[i] = 1.0f;

    Tensor *d_in, *d_w, *d_b;
    conv2d_backward(input, weight, NULL, 2, 0, 2, grad_out, &d_in, &d_w, &d_b);

    TEST_ASSERT_EQUAL_FLOAT(1.0f, d_w->data[0]);

    TEST_ASSERT_EQUAL_FLOAT(1.0f, d_in->data[0]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(grad_out);
    tensor_free(d_in);
    tensor_free(d_w);
}

void test_conv2d_backward_padded(void) {
    uint64_t in_shape[] = {1, 1, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 1.0f;

    uint64_t w_shape[] = {1, 1, 2, 2};
    Tensor *weight = tensor_zeros(w_shape, 4, false);
    for (int i = 0; i < 4; ++i)
        weight->data[i] = 1.0f;

    uint64_t gout_shape[] = {1, 1, 3, 3};
    Tensor *grad_out = tensor_zeros(gout_shape, 4, false);
    for (int i = 0; i < 9; ++i)
        grad_out->data[i] = 1.0f;

    Tensor *d_in, *d_w, *d_b;
    conv2d_backward(input, weight, NULL, 1, 1, 2, grad_out, &d_in, &d_w, &d_b);

    TEST_ASSERT_EQUAL_UINT64(2, d_in->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(2, d_in->shape[3]);

    tensor_free(input);
    tensor_free(weight);
    tensor_free(grad_out);
    tensor_free(d_in);
    tensor_free(d_w);
}

void test_maxpool2d_negative_values(void) {
    uint64_t in_shape[] = {1, 1, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = -10.0f;
    input->data[1] = -5.0f;
    input->data[2] = -20.0f;
    input->data[3] = -8.0f;

    Layer *l = layer_maxpool2d_create(2, 2, 0);
    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(-5.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_maxpool2d_stride_1(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    for (int i = 0; i < 9; ++i)
        input->data[i] = (float)i;

    Layer *l = layer_maxpool2d_create(2, 1, 0);
    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(4.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, out->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(7.0f, out->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(8.0f, out->data[3]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_maxpool2d_non_overlapping(void) {
    uint64_t in_shape[] = {1, 1, 4, 4};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[5] = 100.0f;

    Layer *l = layer_maxpool2d_create(2, 2, 0);
    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(100.0f, out->data[1]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_maxpool2d_padded(void) {
    uint64_t in_shape[] = {1, 1, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 10.0f;

    Layer *l = layer_maxpool2d_create(2, 1, 1);
    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(10.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(10.0f, out->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(10.0f, out->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(10.0f, out->data[3]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_maxpool2d_batch(void) {
    uint64_t in_shape[] = {2, 1, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 1.0f;
    input->data[4] = 2.0f;

    Layer *l = layer_maxpool2d_create(2, 2, 0);
    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(1.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, out->data[1]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_avgpool2d_uniform(void) {
    uint64_t in_shape[] = {1, 1, 4, 4};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    for (int i = 0; i < 16; ++i)
        input->data[i] = 5.0f;

    Layer *l = layer_avgpool2d_create(2, 2, 0);
    Tensor *out = l->forward(l, input, false);

    // Average of 5,5,5,5 is 5.
    TEST_ASSERT_EQUAL_FLOAT(5.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, out->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, out->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, out->data[3]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_avgpool2d_padding_zeros(void) {
    uint64_t in_shape[] = {1, 1, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 4.0f;

    Layer *l = layer_avgpool2d_create(2, 1, 1);
    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(1.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_avgpool2d_stride_gap(void) {
    uint64_t in_shape[] = {1, 1, 4, 4};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 1.0f;
    input->data[1] = 3.0f;
    input->data[4] = 5.0f;
    input->data[5] = 7.0f;

    Layer *l = layer_avgpool2d_create(2, 3, 0);
    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_UINT64(1, out->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(1, out->shape[3]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_avgpool2d_stride_overlap(void) {
    uint64_t in_shape[] = {1, 1, 3, 3};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[4] = 4.0f;

    Layer *l = layer_avgpool2d_create(2, 1, 0);
    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(1.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, out->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, out->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, out->data[3]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_avgpool2d_large(void) {
    uint64_t in_shape[] = {1, 1, 10, 10};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    for (int i = 0; i < 100; ++i)
        input->data[i] = 10.0f;

    Layer *l = layer_avgpool2d_create(5, 5, 0);
    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_UINT64(2, out->shape[2]);
    TEST_ASSERT_EQUAL_FLOAT(10.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_batchnorm2d_training_update(void) {
    uint64_t in_shape[] = {2, 1, 2, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    for (int i = 0; i < 8; ++i)
        input->data[i] = 10.0f;

    Layer *l = layer_batchnorm2d_create(1, 1e-5f, 0.1f);

    Tensor *out = l->forward(l, input, true);

    tensor_free(out);

    Tensor *out_eval = l->forward(l, input, false);

    TEST_ASSERT_FLOAT_WITHIN(1.0f, 9.48f, out_eval->data[0]);

    tensor_free(input);
    tensor_free(out_eval);
    l->free(l);
}

void test_batchnorm2d_eval_fixed(void) {
    uint64_t in_shape[] = {1, 1, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 100.0f;

    Layer *l = layer_batchnorm2d_create(1, 1e-5f, 0.1f);

    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(100.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_batchnorm2d_gamma_beta(void) {
    uint64_t in_shape[] = {1, 1, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 10.0f;

    Layer *l = layer_batchnorm2d_create(1, 1e-5f, 0.1f);
    Tensor **params;
    size_t count;
    l->parameters(l, &params, &count);
    params[0]->data[0] = 2.0f;
    params[1]->data[0] = 5.0f;
    free(params);

    Tensor *out = l->forward(l, input, true);

    TEST_ASSERT_EQUAL_FLOAT(5.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_batchnorm2d_eps_stability(void) {
    uint64_t in_shape[] = {2, 1, 1, 1};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    input->data[0] = 5.0f;
    input->data[1] = 5.0f;

    Layer *l = layer_batchnorm2d_create(1, 1e-5f, 0.1f);
    Tensor *out = l->forward(l, input, true);

    TEST_ASSERT_EQUAL_FLOAT(0.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_batchnorm2d_manual_check(void) {
    uint64_t in_shape[] = {1, 1, 1, 2};
    Tensor *input = tensor_zeros(in_shape, 4, false);
    input->data[0] = 0.0f;
    input->data[1] = 2.0f;

    Layer *l = layer_batchnorm2d_create(1, 0.0f, 0.1f);

    Tensor *out = l->forward(l, input, true);

    TEST_ASSERT_EQUAL_FLOAT(-1.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, out->data[1]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_simplecnn_variable_size(void) {
    uint64_t in_shape[] = {1, 3, 16, 16};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    Layer *l = simple_cnn_create(10);
    Tensor *out = l->forward(l, input, false);

    TEST_ASSERT_EQUAL_UINT64(512, out->shape[1]);

    tensor_free(input);
    tensor_free(out);
    l->free(l);
}

void test_simplecnn_zero_prop(void) {
    uint64_t in_shape[] = {1, 3, 32, 32};
    Tensor *input = tensor_zeros(in_shape, 4, false);

    Layer *l = simple_cnn_create(10);
    Tensor *out = l->forward(l, input, false);

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
    RUN_TEST(test_conv2d_3x3_stride1_pad1);
    RUN_TEST(test_conv2d_kernel_larger_than_input_with_pad);
    RUN_TEST(test_conv2d_batch_size_multi);
    RUN_TEST(test_conv2d_multi_channel);
    RUN_TEST(test_conv2d_bias_value);
    RUN_TEST(test_conv2d_negative_input);
    RUN_TEST(test_conv2d_large_padding);
    RUN_TEST(test_conv2d_rect_input);
    RUN_TEST(test_conv2d_1x1_kernel);
    RUN_TEST(test_conv2d_deep_channels);
    RUN_TEST(test_conv2d_backward_bias_accum);
    RUN_TEST(test_conv2d_backward_grad_input_shape);
    RUN_TEST(test_conv2d_backward_grad_weight_shape);
    RUN_TEST(test_conv2d_backward_strided);
    RUN_TEST(test_conv2d_backward_padded);
    RUN_TEST(test_maxpool2d_negative_values);
    RUN_TEST(test_maxpool2d_stride_1);
    RUN_TEST(test_maxpool2d_non_overlapping);
    RUN_TEST(test_maxpool2d_padded);
    RUN_TEST(test_maxpool2d_batch);
    RUN_TEST(test_avgpool2d_uniform);
    RUN_TEST(test_avgpool2d_padding_zeros);
    RUN_TEST(test_avgpool2d_stride_gap);
    RUN_TEST(test_avgpool2d_stride_overlap);
    RUN_TEST(test_avgpool2d_large);
    RUN_TEST(test_batchnorm2d_training_update);
    RUN_TEST(test_batchnorm2d_eval_fixed);
    RUN_TEST(test_batchnorm2d_gamma_beta);
    RUN_TEST(test_batchnorm2d_eps_stability);
    RUN_TEST(test_batchnorm2d_manual_check);
    RUN_TEST(test_simplecnn_variable_size);
    RUN_TEST(test_simplecnn_zero_prop);
    return UNITY_END();
}
