#include "ops/reductions.h"
#include "ops/reductions_backward.h"
#include "tensor.h"
#include "unity.h"
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#define EPSILON 1e-4f

static void check_tensor_eq(const Tensor *actual, const Tensor *expected) {
    if (actual == NULL || expected == NULL) {
        TEST_FAIL_MESSAGE("One of the tensors is NULL");
    }
    TEST_ASSERT_EQUAL_UINT64(expected->ndim, actual->ndim);
    TEST_ASSERT_EQUAL_UINT64(expected->size, actual->size);

    for (uint64_t i = 0; i < expected->ndim; i++) {
        TEST_ASSERT_EQUAL_UINT64(expected->shape[i], actual->shape[i]);
    }

    for (uint64_t i = 0; i < expected->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected->data[i], actual->data[i]);
    }
}

static Tensor *create_test_tensor(const float32_t *data, const uint64_t *shape, uint64_t ndim, bool requires_grad) {
    Tensor *t = tensor_create(data, shape, ndim, requires_grad);
    return t;
}

void setUp(void) {}
void tearDown(void) {}

void test_sum_backward_1d_all(void) {
    const uint64_t shape[] = {3};
    const float32_t data[] = {1, 2, 3};
    Tensor *x = create_test_tensor(data, shape, 1, true);

    const uint64_t grad_shape[] = {1};
    const float32_t grad_data[] = {1.0f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 0, false);

    Tensor *dx = tensor_sum_backward(grad_out, x, 0, false);

    const float32_t expected_data[] = {1, 1, 1};
    Tensor *expected = create_test_tensor(expected_data, shape, 1, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_sum_backward_2d_dim0(void) {
    const uint64_t shape[] = {2, 3};
    const float32_t data[] = {1, 2, 3, 1, 2, 3};
    Tensor *x = create_test_tensor(data, shape, 2, true);

    const uint64_t grad_shape[] = {3};
    const float32_t grad_data[] = {1, 2, 3};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 1, false);

    Tensor *dx = tensor_sum_backward(grad_out, x, 0, false);

    const float32_t expected_data[] = {1, 2, 3, 1, 2, 3};
    Tensor *expected = create_test_tensor(expected_data, shape, 2, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_sum_backward_2d_dim1(void) {
    const uint64_t shape[] = {2, 3};
    const float32_t data[] = {1, 2, 3, 1, 2, 3};
    Tensor *x = create_test_tensor(data, shape, 2, true);

    const uint64_t grad_shape[] = {2};
    const float32_t grad_data[] = {10, 20};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 1, false);

    Tensor *dx = tensor_sum_backward(grad_out, x, 1, false);

    const float32_t expected_data[] = {10, 10, 10, 20, 20, 20};
    Tensor *expected = create_test_tensor(expected_data, shape, 2, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_sum_backward_2d_keepdims(void) {
    const uint64_t shape[] = {2, 3};
    const float32_t data[] = {1, 2, 3, 1, 2, 3};
    Tensor *x = create_test_tensor(data, shape, 2, true);

    const uint64_t grad_shape[] = {2, 1};
    const float32_t grad_data[] = {5, 6};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 2, false);

    Tensor *dx = tensor_sum_backward(grad_out, x, 1, true);

    const float32_t expected_data[] = {5, 5, 5, 6, 6, 6};
    Tensor *expected = create_test_tensor(expected_data, shape, 2, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_sum_backward_3d_dim1(void) {
    const uint64_t shape[] = {2, 2, 2};
    Tensor *x = tensor_zeros(shape, 3, true);

    const uint64_t grad_shape[] = {2, 2};
    const float32_t grad_data[] = {1, 2, 3, 4};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 2, false);

    Tensor *dx = tensor_sum_backward(grad_out, x, 1, false);

    const float32_t expected_data[] = {1, 2, 1, 2, 3, 4, 3, 4};
    Tensor *expected = create_test_tensor(expected_data, shape, 3, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_sum_backward_neg_dim(void) {
    const uint64_t shape[] = {2, 3};
    const float32_t data[] = {1, 2, 3, 1, 2, 3};
    Tensor *x = create_test_tensor(data, shape, 2, true);

    const uint64_t grad_shape[] = {2};
    const float32_t grad_data[] = {1, 2};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 1, false);

    Tensor *dx = tensor_sum_backward(grad_out, x, -1, false);

    const float32_t expected_data[] = {1, 1, 1, 2, 2, 2};
    Tensor *expected = create_test_tensor(expected_data, shape, 2, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_sum_backward_ones(void) {
    const uint64_t shape[] = {2, 2};
    const float32_t data[] = {1, 2, 3, 4};
    Tensor *x = create_test_tensor(data, shape, 2, true);

    const uint64_t grad_shape[] = {2};
    const float32_t grad_data[] = {1, 1};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 1, false);

    Tensor *dx = tensor_sum_backward(grad_out, x, 1, false);

    const float32_t expected_data[] = {1, 1, 1, 1};
    Tensor *expected = create_test_tensor(expected_data, shape, 2, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_sum_backward_random_grad(void) {
    const uint64_t shape[] = {3};
    Tensor *x = tensor_zeros(shape, 1, true);

    const uint64_t grad_shape[] = {1};
    const float32_t grad_data[] = {0.5f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 0, false);

    Tensor *dx = tensor_sum_backward(grad_out, x, 0, false);

    const float32_t expected_data[] = {0.5f, 0.5f, 0.5f};
    Tensor *expected = create_test_tensor(expected_data, shape, 1, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_sum_backward_zero_grad(void) {
    const uint64_t shape[] = {2};
    Tensor *x = tensor_zeros(shape, 1, true);

    const uint64_t grad_shape[] = {1};
    const float32_t grad_data[] = {0.0f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 0, false);

    Tensor *dx = tensor_sum_backward(grad_out, x, 0, false);

    const float32_t expected_data[] = {0.0f, 0.0f};
    Tensor *expected = create_test_tensor(expected_data, shape, 1, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_sum_backward_single_element(void) {
    const uint64_t shape[] = {1};
    Tensor *x = tensor_zeros(shape, 1, true);

    const uint64_t grad_shape[] = {1};
    const float32_t grad_data[] = {7.0f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 0, false);

    Tensor *dx = tensor_sum_backward(grad_out, x, 0, false);

    const float32_t expected_data[] = {7.0f};
    Tensor *expected = create_test_tensor(expected_data, shape, 1, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_mean_backward_1d(void) {
    const uint64_t shape[] = {4};
    Tensor *x = tensor_zeros(shape, 1, true);

    const uint64_t grad_shape[] = {1};
    const float32_t grad_data[] = {1.0f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 0, false);

    Tensor *dx = tensor_mean_backward(grad_out, x, 0, false);

    const float32_t expected_data[] = {0.25f, 0.25f, 0.25f, 0.25f};
    Tensor *expected = create_test_tensor(expected_data, shape, 1, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_mean_backward_2d_dim0(void) {
    const uint64_t shape[] = {2, 3};
    const float32_t data[] = {1, 2, 3, 1, 2, 3};
    Tensor *x = create_test_tensor(data, shape, 2, true);

    const uint64_t grad_shape[] = {3};
    const float32_t grad_data[] = {1, 2, 3};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 1, false);

    Tensor *dx = tensor_mean_backward(grad_out, x, 0, false);

    const float32_t expected_data[] = {0.5f, 1.0f, 1.5f, 0.5f, 1.0f, 1.5f};
    Tensor *expected = create_test_tensor(expected_data, shape, 2, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_mean_backward_2d_dim1(void) {
    const uint64_t shape[] = {2, 3};
    Tensor *x = tensor_zeros(shape, 2, true);

    const uint64_t grad_shape[] = {2};
    const float32_t grad_data[] = {3, 6};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 1, false);

    Tensor *dx = tensor_mean_backward(grad_out, x, 1, false);

    const float32_t expected_data[] = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
    Tensor *expected = create_test_tensor(expected_data, shape, 2, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_mean_backward_2d_keepdims(void) {
    const uint64_t shape[] = {2, 2};
    Tensor *x = tensor_zeros(shape, 2, true);

    const uint64_t grad_shape[] = {1, 2};
    const float32_t grad_data[] = {4, 8};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 2, false);

    Tensor *dx = tensor_mean_backward(grad_out, x, 0, true);

    const float32_t expected_data[] = {2.0f, 4.0f, 2.0f, 4.0f};
    Tensor *expected = create_test_tensor(expected_data, shape, 2, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_mean_backward_3d_dim2(void) {
    const uint64_t shape[] = {1, 1, 2};
    Tensor *x = tensor_zeros(shape, 3, true);

    const uint64_t grad_shape[] = {1, 1};
    const float32_t grad_data[] = {10.0f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 2, false);

    Tensor *dx = tensor_mean_backward(grad_out, x, 2, false);

    const float32_t expected_data[] = {5.0f, 5.0f};
    Tensor *expected = create_test_tensor(expected_data, shape, 3, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_mean_backward_neg_dim(void) {
    const uint64_t shape[] = {4};
    Tensor *x = tensor_zeros(shape, 1, true);

    const uint64_t grad_shape[] = {1};
    const float32_t grad_data[] = {4.0f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 0, false);

    Tensor *dx = tensor_mean_backward(grad_out, x, -1, false);

    const float32_t expected_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
    Tensor *expected = create_test_tensor(expected_data, shape, 1, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_mean_backward_large_n(void) {
    const uint64_t shape[] = {2, 5};
    Tensor *x = tensor_zeros(shape, 2, true);

    const uint64_t grad_shape[] = {2};
    const float32_t grad_data[] = {10, 20};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 1, false);

    Tensor *dx = tensor_mean_backward(grad_out, x, 1, false);

    const float32_t expected_data[] = {2, 2, 2, 2, 2, 4, 4, 4, 4, 4};
    Tensor *expected = create_test_tensor(expected_data, shape, 2, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_mean_backward_random_grad(void) {
    const uint64_t shape[] = {2};
    Tensor *x = tensor_zeros(shape, 1, true);

    const uint64_t grad_shape[] = {1};
    const float32_t grad_data[] = {3.14f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 0, false);

    Tensor *dx = tensor_mean_backward(grad_out, x, 0, false);

    const float32_t expected_data[] = {1.57f, 1.57f};
    Tensor *expected = create_test_tensor(expected_data, shape, 1, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_mean_backward_zero_grad(void) {
    const uint64_t shape[] = {2};
    Tensor *x = tensor_zeros(shape, 1, true);

    const uint64_t grad_shape[] = {1};
    const float32_t grad_data[] = {0.0f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 0, false);

    Tensor *dx = tensor_mean_backward(grad_out, x, 0, false);

    const float32_t expected_data[] = {0.0f, 0.0f};
    Tensor *expected = create_test_tensor(expected_data, shape, 1, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_mean_backward_single_element(void) {
    const uint64_t shape[] = {1};
    Tensor *x = tensor_zeros(shape, 1, true);

    const uint64_t grad_shape[] = {1};
    const float32_t grad_data[] = {8.0f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 0, false);

    Tensor *dx = tensor_mean_backward(grad_out, x, 0, false);

    const float32_t expected_data[] = {8.0f};
    Tensor *expected = create_test_tensor(expected_data, shape, 1, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_max_backward_1d_unique(void) {
    const uint64_t shape[] = {3};
    const float32_t data[] = {1, 3, 2};
    Tensor *x = create_test_tensor(data, shape, 1, true);

    Tensor *out = tensor_max(x, 0, false);

    const uint64_t grad_shape[] = {1};
    const float32_t grad_data[] = {5.0f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 0, false);

    Tensor *dx = tensor_max_backward(grad_out, x, out, 0, false);

    const float32_t expected_data[] = {0, 5, 0};
    Tensor *expected = create_test_tensor(expected_data, shape, 1, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(out);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_max_backward_1d_duplicate(void) {
    const uint64_t shape[] = {4};
    const float32_t data[] = {1, 3, 2, 3};
    Tensor *x = create_test_tensor(data, shape, 1, true);

    Tensor *out = tensor_max(x, 0, false);

    const uint64_t grad_shape[] = {1};
    const float32_t grad_data[] = {1.0f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 0, false);

    Tensor *dx = tensor_max_backward(grad_out, x, out, 0, false);

    const float32_t expected_data[] = {0, 1, 0, 1};
    Tensor *expected = create_test_tensor(expected_data, shape, 1, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(out);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_max_backward_2d_dim0(void) {
    const uint64_t shape[] = {2, 2};
    const float32_t data[] = {1, 4, 3, 2};
    Tensor *x = create_test_tensor(data, shape, 2, true);

    Tensor *out = tensor_max(x, 0, false);

    const uint64_t grad_shape[] = {2};
    const float32_t grad_data[] = {10, 10};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 1, false);

    Tensor *dx = tensor_max_backward(grad_out, x, out, 0, false);

    const float32_t expected_data[] = {0, 10, 10, 0};
    Tensor *expected = create_test_tensor(expected_data, shape, 2, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(out);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_max_backward_2d_dim1(void) {
    const uint64_t shape[] = {2, 2};
    const float32_t data[] = {1, 4, 3, 2};
    Tensor *x = create_test_tensor(data, shape, 2, true);

    Tensor *out = tensor_max(x, 1, false);

    const uint64_t grad_shape[] = {2};
    const float32_t grad_data[] = {10, 10};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 1, false);

    Tensor *dx = tensor_max_backward(grad_out, x, out, 1, false);

    const float32_t expected_data[] = {0, 10, 10, 0};
    Tensor *expected = create_test_tensor(expected_data, shape, 2, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(out);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_max_backward_keepdims(void) {
    const uint64_t shape[] = {2};
    const float32_t data[] = {1, 2};
    Tensor *x = create_test_tensor(data, shape, 1, true);

    Tensor *out = tensor_max(x, 0, true);

    const uint64_t grad_shape[] = {1};
    const float32_t grad_data[] = {1.0f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 1, false);

    Tensor *dx = tensor_max_backward(grad_out, x, out, 0, true);

    const float32_t expected_data[] = {0, 1};
    Tensor *expected = create_test_tensor(expected_data, shape, 1, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(out);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_max_backward_neg_dim(void) {
    const uint64_t shape[] = {2};
    const float32_t data[] = {5, 1};
    Tensor *x = create_test_tensor(data, shape, 1, true);

    Tensor *out = tensor_max(x, -1, false);

    const uint64_t grad_shape[] = {1};
    const float32_t grad_data[] = {1.0f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 0, false);

    Tensor *dx = tensor_max_backward(grad_out, x, out, -1, false);

    const float32_t expected_data[] = {1, 0};
    Tensor *expected = create_test_tensor(expected_data, shape, 1, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(out);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_max_backward_all_same(void) {
    const uint64_t shape[] = {3};
    const float32_t data[] = {2, 2, 2};
    Tensor *x = create_test_tensor(data, shape, 1, true);

    Tensor *out = tensor_max(x, 0, false);

    const uint64_t grad_shape[] = {1};
    const float32_t grad_data[] = {6.0f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 0, false);

    Tensor *dx = tensor_max_backward(grad_out, x, out, 0, false);

    const float32_t expected_data[] = {6, 6, 6};
    Tensor *expected = create_test_tensor(expected_data, shape, 1, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(out);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_max_backward_negative_values(void) {
    const uint64_t shape[] = {3};
    const float32_t data[] = {-5, -1, -3};
    Tensor *x = create_test_tensor(data, shape, 1, true);

    Tensor *out = tensor_max(x, 0, false);

    const uint64_t grad_shape[] = {1};
    const float32_t grad_data[] = {1.0f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 0, false);

    Tensor *dx = tensor_max_backward(grad_out, x, out, 0, false);

    const float32_t expected_data[] = {0, 1, 0};
    Tensor *expected = create_test_tensor(expected_data, shape, 1, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(out);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_max_backward_3d_complex(void) {
    const uint64_t shape[] = {1, 2, 2};
    const float32_t data[] = {1, 2, 3, 4};
    Tensor *x = create_test_tensor(data, shape, 3, true);

    Tensor *out = tensor_max(x, 2, false);
    const uint64_t grad_shape[] = {1, 2};
    const float32_t grad_data[] = {10, 20};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 2, false);

    Tensor *dx = tensor_max_backward(grad_out, x, out, 2, false);

    const float32_t expected_data[] = {0, 10, 0, 20};
    Tensor *expected = create_test_tensor(expected_data, shape, 3, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(out);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

void test_max_backward_zero_grad(void) {
    const uint64_t shape[] = {2};
    const float32_t data[] = {1, 2};
    Tensor *x = create_test_tensor(data, shape, 1, true);

    Tensor *out = tensor_max(x, 0, false);

    const uint64_t grad_shape[] = {1};
    const float32_t grad_data[] = {0.0f};
    Tensor *grad_out = create_test_tensor(grad_data, grad_shape, 0, false);

    Tensor *dx = tensor_max_backward(grad_out, x, out, 0, false);

    const float32_t expected_data[] = {0, 0};
    Tensor *expected = create_test_tensor(expected_data, shape, 1, false);

    check_tensor_eq(dx, expected);

    tensor_free(x);
    tensor_free(out);
    tensor_free(grad_out);
    tensor_free(dx);
    tensor_free(expected);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_sum_backward_1d_all);
    RUN_TEST(test_sum_backward_2d_dim0);
    RUN_TEST(test_sum_backward_2d_dim1);
    RUN_TEST(test_sum_backward_2d_keepdims);
    RUN_TEST(test_sum_backward_3d_dim1);
    RUN_TEST(test_sum_backward_neg_dim);
    RUN_TEST(test_sum_backward_ones);
    RUN_TEST(test_sum_backward_random_grad);
    RUN_TEST(test_sum_backward_zero_grad);
    RUN_TEST(test_sum_backward_single_element);
    RUN_TEST(test_mean_backward_1d);
    RUN_TEST(test_mean_backward_2d_dim0);
    RUN_TEST(test_mean_backward_2d_dim1);
    RUN_TEST(test_mean_backward_2d_keepdims);
    RUN_TEST(test_mean_backward_3d_dim2);
    RUN_TEST(test_mean_backward_neg_dim);
    RUN_TEST(test_mean_backward_large_n);
    RUN_TEST(test_mean_backward_random_grad);
    RUN_TEST(test_mean_backward_zero_grad);
    RUN_TEST(test_mean_backward_single_element);
    RUN_TEST(test_max_backward_1d_unique);
    RUN_TEST(test_max_backward_1d_duplicate);
    RUN_TEST(test_max_backward_2d_dim0);
    RUN_TEST(test_max_backward_2d_dim1);
    RUN_TEST(test_max_backward_keepdims);
    RUN_TEST(test_max_backward_neg_dim);
    RUN_TEST(test_max_backward_all_same);
    RUN_TEST(test_max_backward_negative_values);
    RUN_TEST(test_max_backward_3d_complex);
    RUN_TEST(test_max_backward_zero_grad);
    return UNITY_END();
}
