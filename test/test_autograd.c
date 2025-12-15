#include "autograd.h"
#include "ops/arithmetic.h"
#include "tensor.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_add_backward_simple(void) {
    uint64_t shape[] = {};
    float32_t x_data = 2.0f;
    float32_t y_data = 3.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_create(&y_data, shape, 0, true);

    Tensor *z = tensor_add(x, y);

    backward(z);

    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_NOT_NULL(y->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, y->grad->data[0]);

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
}

void test_add_backward_chain(void) {
    uint64_t shape[] = {};
    float32_t x_data = 2.0f;
    float32_t y_data = 3.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_create(&y_data, shape, 0, true);

    Tensor *z = tensor_add(x, y);
    Tensor *w = tensor_add(z, x);

    backward(w);

    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_NOT_NULL(y->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, y->grad->data[0]);

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
    tensor_release(w);
}

void test_add_backward_diamond(void) {
    uint64_t shape[] = {};
    float32_t x_data = 2.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);

    Tensor *z = tensor_add(x, x);

    backward(z);

    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, x->grad->data[0]);

    tensor_release(x);
    tensor_release(z);
}

void test_mul_backward_simple(void) {
    uint64_t shape[] = {};
    float32_t x_data = 3.0f;
    float32_t y_data = 4.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_create(&y_data, shape, 0, true);

    Tensor *z = tensor_mul(x, y);

    backward(z);

    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_NOT_NULL(y->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 4.0f, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 3.0f, y->grad->data[0]);

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
}

void test_mul_add_chain(void) {
    uint64_t shape[] = {};
    float32_t x_data = 2.0f;
    float32_t y_data = 3.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_create(&y_data, shape, 0, true);

    Tensor *z = tensor_mul(x, y);
    Tensor *w = tensor_add(z, x);

    backward(w);

    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_NOT_NULL(y->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 4.0f, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, y->grad->data[0]);

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
    tensor_release(w);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_add_backward_simple);
    RUN_TEST(test_add_backward_chain);
    RUN_TEST(test_add_backward_diamond);
    RUN_TEST(test_mul_backward_simple);
    RUN_TEST(test_mul_add_chain);
    return UNITY_END();
}
