#include "autograd.h"
#include "ops/arithmetic.h"
#include "ops/reductions.h"
#include "tensor.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_sum_backward_simple_1d(void) {
    uint64_t shape[] = {3};
    float32_t data[] = {1.0f, 2.0f, 3.0f};

    Tensor *x = tensor_create(data, shape, 1, true);
    Tensor *y = tensor_sum(x, 0, false);

    backward(y);

    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[2]);

    tensor_release(x);
    tensor_release(y);
}

void test_sum_backward_2d_dim0(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_sum(x, 0, false);

    Tensor *scalar = tensor_sum(y, 0, false);
    backward(scalar);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(scalar);
}

void test_sum_backward_2d_dim1(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_sum(x, 1, false);

    Tensor *scalar = tensor_sum(y, 0, false);
    backward(scalar);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(scalar);
}

void test_sum_backward_keepdims(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_sum(x, 0, true);

    Tensor *temp = tensor_sum(y, 0, false);
    Tensor *scalar = tensor_sum(temp, 0, false);
    backward(scalar);

    TEST_ASSERT_NOT_NULL(x->grad);

    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(temp);
    tensor_release(scalar);
}

void test_mean_backward_simple_1d(void) {
    uint64_t shape[] = {3};
    float32_t data[] = {1.0f, 2.0f, 3.0f};

    Tensor *x = tensor_create(data, shape, 1, true);
    Tensor *y = tensor_mean(x, 0, false);

    backward(y);

    TEST_ASSERT_NOT_NULL(x->grad);
    float32_t expected_grad = 1.0f / 3.0f;
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_grad, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_grad, x->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_grad, x->grad->data[2]);

    tensor_release(x);
    tensor_release(y);
}

void test_mean_backward_2d_dim0(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_mean(x, 0, false);

    Tensor *scalar = tensor_sum(y, 0, false);
    backward(scalar);

    TEST_ASSERT_NOT_NULL(x->grad);

    float32_t expected_grad = 1.0f / 2.0f;
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_grad, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(scalar);
}

void test_mean_backward_2d_dim1(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_mean(x, 1, false);

    Tensor *scalar = tensor_sum(y, 0, false);
    backward(scalar);

    TEST_ASSERT_NOT_NULL(x->grad);

    float32_t expected_grad = 1.0f / 3.0f;
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_grad, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(scalar);
}

void test_mean_backward_keepdims(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_mean(x, 0, true);

    Tensor *temp = tensor_sum(y, 0, false);
    Tensor *scalar = tensor_sum(temp, 0, false);
    backward(scalar);

    TEST_ASSERT_NOT_NULL(x->grad);

    float32_t expected_grad = 1.0f / 2.0f;
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_grad, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(temp);
    tensor_release(scalar);
}

void test_max_backward_simple_1d(void) {
    uint64_t shape[] = {3};
    float32_t data[] = {1.0f, 3.0f, 2.0f};

    Tensor *x = tensor_create(data, shape, 1, true);
    Tensor *y = tensor_max(x, 0, false);

    backward(y);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[2]);

    tensor_release(x);
    tensor_release(y);
}

void test_max_backward_2d_dim0(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_max(x, 0, false);

    Tensor *scalar = tensor_sum(y, 0, false);
    backward(scalar);

    TEST_ASSERT_NOT_NULL(x->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[3]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[4]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[5]);

    tensor_release(x);
    tensor_release(y);
    tensor_release(scalar);
}

void test_max_backward_2d_dim1(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 3.0f, 2.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_max(x, 1, false);

    Tensor *scalar = tensor_sum(y, 0, false);
    backward(scalar);

    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[3]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[4]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[5]);

    tensor_release(x);
    tensor_release(y);
    tensor_release(scalar);
}

void test_max_backward_keepdims(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_max(x, 0, true);

    Tensor *temp = tensor_sum(y, 0, false);
    Tensor *scalar = tensor_sum(temp, 0, false);
    backward(scalar);

    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[3]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[4]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[5]);

    tensor_release(x);
    tensor_release(y);
    tensor_release(temp);
    tensor_release(scalar);
}

void test_sum_backward_chain(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_sum(x, 0, false);
    Tensor *z = tensor_sum(y, 0, false);

    backward(z);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
}

void test_mean_backward_chain(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_mean(x, 0, false);
    Tensor *z = tensor_sum(y, 0, false);

    backward(z);

    TEST_ASSERT_NOT_NULL(x->grad);
    float32_t expected_grad = 1.0f / 2.0f;
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_grad, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
}

void test_sum_add_chain(void) {
    uint64_t shape[] = {3};
    float32_t data[] = {1.0f, 2.0f, 3.0f};

    Tensor *x = tensor_create(data, shape, 1, true);
    Tensor *y = tensor_sum(x, 0, false);
    Tensor *z = tensor_add(y, y);

    backward(z);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
}

void test_max_backward_tie(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {3.0f, 2.0f, 3.0f, 3.0f, 1.0f, 2.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_max(x, 1, false);

    Tensor *scalar = tensor_sum(y, 0, false);
    backward(scalar);

    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[3]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[4]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, x->grad->data[5]);

    tensor_release(x);
    tensor_release(y);
    tensor_release(scalar);
}

void test_sum_backward_negative_dim(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_sum(x, -1, false);

    Tensor *scalar = tensor_sum(y, 0, false);
    backward(scalar);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(scalar);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_sum_backward_simple_1d);
    RUN_TEST(test_sum_backward_2d_dim0);
    RUN_TEST(test_sum_backward_2d_dim1);
    RUN_TEST(test_sum_backward_keepdims);
    RUN_TEST(test_mean_backward_simple_1d);
    RUN_TEST(test_mean_backward_2d_dim0);
    RUN_TEST(test_mean_backward_2d_dim1);
    RUN_TEST(test_mean_backward_keepdims);
    RUN_TEST(test_max_backward_simple_1d);
    RUN_TEST(test_max_backward_2d_dim0);
    RUN_TEST(test_max_backward_2d_dim1);
    RUN_TEST(test_max_backward_keepdims);
    RUN_TEST(test_sum_backward_chain);
    RUN_TEST(test_mean_backward_chain);
    RUN_TEST(test_sum_add_chain);
    RUN_TEST(test_max_backward_tie);
    RUN_TEST(test_sum_backward_negative_dim);
    return UNITY_END();
}
