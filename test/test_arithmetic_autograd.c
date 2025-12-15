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

void test_sub_backward_simple(void) {
    uint64_t shape[] = {};
    float32_t x_data = 5.0f;
    float32_t y_data = 2.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_create(&y_data, shape, 0, true);

    Tensor *z = tensor_sub(x, y);

    backward(z);

    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_NOT_NULL(y->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -1.0f, y->grad->data[0]);

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
}

void test_div_backward_simple(void) {
    uint64_t shape[] = {};
    float32_t x_data = 6.0f;
    float32_t y_data = 3.0f;

    Tensor *x = tensor_create(&x_data, shape, 0, true);
    Tensor *y = tensor_create(&y_data, shape, 0, true);

    Tensor *z = tensor_div(x, y);

    backward(z);

    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_NOT_NULL(y->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f / 3.0f, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -2.0f / 3.0f, y->grad->data[0]);

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
}

void test_matmul_forward_simple(void) {
    uint64_t shape_a[] = {1, 2};
    float32_t data_a[] = {1.0f, 2.0f};

    uint64_t shape_b[] = {2, 1};
    float32_t data_b[] = {3.0f, 4.0f};

    Tensor *a = tensor_create(data_a, shape_a, 2, true);
    Tensor *b = tensor_create(data_b, shape_b, 2, true);

    Tensor *c = tensor_matmul(a, b);

    TEST_ASSERT_EQUAL_UINT64(2, c->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, c->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(1, c->shape[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 11.0f, c->data[0]);

    tensor_release(a);
    tensor_release(b);
    tensor_release(c);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_add_backward_simple);
    RUN_TEST(test_add_backward_chain);
    RUN_TEST(test_add_backward_diamond);
    RUN_TEST(test_mul_backward_simple);
    RUN_TEST(test_mul_add_chain);
    RUN_TEST(test_sub_backward_simple);
    RUN_TEST(test_div_backward_simple);
    RUN_TEST(test_matmul_forward_simple);
    return UNITY_END();
}
