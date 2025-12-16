#include "autograd.h"
#include "ops/activations.h"
#include "ops/arithmetic.h"
#include "ops/reductions.h"
#include "tensor.h"
#include "unity.h"
#include <math.h>

void setUp(void) {}
void tearDown(void) {}

void test_sigmoid_backward_simple(void) {
    uint64_t shape[] = {};
    float32_t x_data = 0.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_sigmoid(x);

    backward(y);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.25f, x->grad->data[0]);

    tensor_release(x);
    tensor_release(y);
}

void test_sigmoid_backward_chain(void) {
    uint64_t shape[] = {};
    float32_t x_data = 1.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_sigmoid(x);
    Tensor *z = tensor_mul(y, x);

    backward(z);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.9277f, x->grad->data[0]);

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
}

void test_relu_backward_simple(void) {
    uint64_t shape[] = {};
    float32_t x_data = 2.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_relu(x);

    backward(y);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[0]);

    tensor_release(x);
    tensor_release(y);
}

void test_relu_backward_negative(void) {
    uint64_t shape[] = {};
    float32_t x_data = -2.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_relu(x);

    backward(y);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[0]);

    tensor_release(x);
    tensor_release(y);
}

void test_relu_backward_zero(void) {
    uint64_t shape[] = {};
    float32_t x_data = 0.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_relu(x);

    backward(y);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[0]);

    tensor_release(x);
    tensor_release(y);
}

void test_tanh_backward_simple(void) {
    uint64_t shape[] = {};
    float32_t x_data = 0.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_tanh(x);

    backward(y);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[0]);

    tensor_release(x);
    tensor_release(y);
}

void test_tanh_backward_nonzero(void) {
    uint64_t shape[] = {};
    float32_t x_data = 1.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_tanh(x);

    backward(y);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.420f, x->grad->data[0]);

    tensor_release(x);
    tensor_release(y);
}

void test_gelu_backward_simple(void) {
    uint64_t shape[] = {};
    float32_t x_data = 0.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_gelu(x);

    backward(y);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.5f, x->grad->data[0]);

    tensor_release(x);
    tensor_release(y);
}

void test_gelu_backward_nonzero(void) {
    uint64_t shape[] = {};
    float32_t x_data = 1.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_gelu(x);

    backward(y);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_TRUE(x->grad->data[0] > 0.5f);

    tensor_release(x);
    tensor_release(y);
}

void test_softmax_backward_simple(void) {
    float32_t data[] = {1.0f, 2.0f, 3.0f};
    uint64_t shape[] = {3};

    Tensor *x = tensor_create(data, shape, 1, true);
    Tensor *y = tensor_softmax(x, 0);
    Tensor *loss = tensor_sum(y, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);

    float32_t grad_sum = 0.0f;
    for (uint64_t i = 0; i < 3; i++) {
        grad_sum += x->grad->data[i];
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, grad_sum);

    tensor_release(x);
    tensor_release(y);
    tensor_release(loss);
}

void test_sigmoid_backward_vector(void) {
    float32_t data[] = {0.0f, 1.0f, -1.0f};
    uint64_t shape[] = {3};

    Tensor *x = tensor_create(data, shape, 1, true);
    Tensor *y = tensor_sigmoid(x);
    Tensor *loss = tensor_sum(y, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.25f, x->grad->data[0]);

    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.1966f, x->grad->data[1]);

    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.1966f, x->grad->data[2]);

    tensor_release(x);
    tensor_release(y);
    tensor_release(loss);
}

void test_relu_backward_vector(void) {
    float32_t data[] = {-1.0f, 0.0f, 1.0f, 2.0f};
    uint64_t shape[] = {4};

    Tensor *x = tensor_create(data, shape, 1, true);
    Tensor *y = tensor_relu(x);
    Tensor *loss = tensor_sum(y, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[1]);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[3]);

    tensor_release(x);
    tensor_release(y);
    tensor_release(loss);
}

void test_activations_chain(void) {
    uint64_t shape[] = {};
    float32_t x_data = 1.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_relu(x);
    Tensor *z = tensor_sigmoid(y);
    Tensor *w = tensor_tanh(z);

    backward(w);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_TRUE(x->grad->data[0] != 0.0f);

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
    tensor_release(w);
}

void test_sigmoid_mul_chain(void) {
    uint64_t shape[] = {};
    float32_t x_data = 2.0f;
    float32_t w_data = 3.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *w = tensor_create(&w_data, shape, 0, true);
    Tensor *y = tensor_sigmoid(x);
    Tensor *z = tensor_mul(y, w);

    backward(z);

    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_NOT_NULL(w->grad);

    float32_t sigmoid_x = 1.0f / (1.0f + expf(-2.0f));
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 3.0f * sigmoid_x * (1.0f - sigmoid_x), x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, sigmoid_x, w->grad->data[0]);

    tensor_release(x);
    tensor_release(w);
    tensor_release(y);
    tensor_release(z);
}

void test_softmax_backward_2d(void) {
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint64_t shape[] = {2, 2};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_softmax(x, 1);
    Tensor *sum1 = tensor_sum(y, 0, false);
    Tensor *loss = tensor_sum(sum1, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);

    for (uint64_t i = 0; i < 4; i++) {
        TEST_ASSERT_TRUE(!isnan(x->grad->data[i]) && !isinf(x->grad->data[i]));
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(sum1);
    tensor_release(loss);
}

void test_gelu_backward_negative(void) {
    uint64_t shape[] = {};
    float32_t x_data = -1.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_gelu(x);

    backward(y);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_TRUE(x->grad->data[0] < 0.0f);

    tensor_release(x);
    tensor_release(y);
}

void test_tanh_backward_chain_with_add(void) {
    uint64_t shape[] = {};
    float32_t x_data = 0.5f;
    float32_t b_data = 1.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *b = tensor_create(&b_data, shape, 0, true);
    Tensor *y = tensor_tanh(x);
    Tensor *z = tensor_add(y, b);

    backward(z);

    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_NOT_NULL(b->grad);

    float32_t tanh_x = tanhf(0.5f);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f - tanh_x * tanh_x, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, b->grad->data[0]);

    tensor_release(x);
    tensor_release(b);
    tensor_release(y);
    tensor_release(z);
}

void test_sigmoid_backward_no_grad(void) {
    uint64_t shape[] = {};
    float32_t x_data = 1.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, false);
    Tensor *y = tensor_sigmoid(x);

    TEST_ASSERT_FALSE(y->requires_grad);

    tensor_release(x);
    tensor_release(y);
}

void test_relu_backward_requires_grad(void) {
    uint64_t shape[] = {};
    float32_t x_data = 1.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_relu(x);

    TEST_ASSERT_TRUE(y->requires_grad);
    TEST_ASSERT_NOT_NULL(y->grad_fn);

    tensor_release(x);
    tensor_release(y);
}

void test_relu_backward_large_negative(void) {
    uint64_t shape[] = {};
    float32_t val = -1e5f;
    Tensor *x = tensor_create(&val, shape, 0, true);
    Tensor *y = tensor_relu(x);
    backward(y);
    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[0]);
    tensor_release(x);
    tensor_release(y);
}

void test_relu_backward_large_positive(void) {
    uint64_t shape[] = {};
    float32_t val = 1e5f;
    Tensor *x = tensor_create(&val, shape, 0, true);
    Tensor *y = tensor_relu(x);
    backward(y);
    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[0]);
    tensor_release(x);
    tensor_release(y);
}

void test_gelu_backward_large_value(void) {
    uint64_t shape[] = {};
    float32_t val = 10.0f;
    Tensor *x = tensor_create(&val, shape, 0, true);
    Tensor *y = tensor_gelu(x);
    backward(y);
    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f, x->grad->data[0]);
    tensor_release(x);
    tensor_release(y);
}

void test_sigmoid_backward_large_value(void) {
    uint64_t shape[] = {};
    float32_t val = 10.0f;
    Tensor *x = tensor_create(&val, shape, 0, true);
    Tensor *y = tensor_sigmoid(x);
    backward(y);
    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, x->grad->data[0]);
    tensor_release(x);
    tensor_release(y);
}

void test_tanh_backward_large_value(void) {
    uint64_t shape[] = {};
    float32_t val = 10.0f;
    Tensor *x = tensor_create(&val, shape, 0, true);
    Tensor *y = tensor_tanh(x);
    backward(y);
    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, x->grad->data[0]);
    tensor_release(x);
    tensor_release(y);
}

void test_multi_layer_chain(void) {
    uint64_t shape[] = {1};
    float32_t val = 0.5f;
    Tensor *x = tensor_create(&val, shape, 1, true);
    Tensor *y = tensor_relu(x);
    Tensor *z = tensor_sigmoid(y);
    Tensor *loss = tensor_sum(z, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.235003f, x->grad->data[0]);

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
    tensor_release(loss);
}

void test_softmax_dim_check(void) {
    float32_t data[] = {1.0f, 2.0f};
    uint64_t shape[] = {1, 2};
    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_softmax(x, 1);
    Tensor *sum1 = tensor_sum(y, 1, false);
    Tensor *loss = tensor_sum(sum1, 0, false);

    backward(loss);
    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, x->grad->data[1]);

    tensor_release(x);
    tensor_release(y);
    tensor_release(sum1);
    tensor_release(loss);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_sigmoid_backward_simple);
    RUN_TEST(test_sigmoid_backward_chain);
    RUN_TEST(test_relu_backward_simple);
    RUN_TEST(test_relu_backward_negative);
    RUN_TEST(test_relu_backward_zero);
    RUN_TEST(test_tanh_backward_simple);
    RUN_TEST(test_tanh_backward_nonzero);
    RUN_TEST(test_gelu_backward_simple);
    RUN_TEST(test_gelu_backward_nonzero);
    RUN_TEST(test_softmax_backward_simple);
    RUN_TEST(test_sigmoid_backward_vector);
    RUN_TEST(test_relu_backward_vector);
    RUN_TEST(test_activations_chain);
    RUN_TEST(test_sigmoid_mul_chain);
    RUN_TEST(test_softmax_backward_2d);
    RUN_TEST(test_gelu_backward_negative);
    RUN_TEST(test_tanh_backward_chain_with_add);
    RUN_TEST(test_sigmoid_backward_no_grad);
    RUN_TEST(test_relu_backward_requires_grad);
    RUN_TEST(test_relu_backward_large_negative);
    RUN_TEST(test_relu_backward_large_positive);
    RUN_TEST(test_gelu_backward_large_value);
    RUN_TEST(test_sigmoid_backward_large_value);
    RUN_TEST(test_tanh_backward_large_value);
    RUN_TEST(test_multi_layer_chain);
    RUN_TEST(test_softmax_dim_check);
    return UNITY_END();
}
