#include "tensor.h"
#include "unity.h"
#include <float.h>
#include <stdlib.h>
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

void test_tensor_create_scalar(void) {
    float32_t val = 42.0f;
    Tensor *t = tensor_create(&val, NULL, 0, false);

    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(0, t->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, t->size);
    TEST_ASSERT_EQUAL_FLOAT(42.0f, t->data[0]);
    TEST_ASSERT_FALSE(t->requires_grad);
    TEST_ASSERT_NULL(t->grad);
    TEST_ASSERT_NULL(t->shape);

    tensor_free(t);
}

void test_tensor_create_scalar_default_zero(void) {
    Tensor *t = tensor_create(NULL, NULL, 0, true);
    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(0, t->ndim);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, t->data[0]);
    TEST_ASSERT_TRUE(t->requires_grad);
    tensor_free(t);
}

void test_tensor_create_vector(void) {
    float32_t data[] = {1.0f, 2.0f, 3.0f};
    uint64_t shape[] = {3};
    Tensor *t = tensor_create(data, shape, 1, false);

    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(1, t->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, t->size);
    TEST_ASSERT_EQUAL_UINT64(3, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(1, t->strides[0]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, t->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, t->data[2]);

    tensor_free(t);
}

void test_tensor_create_matrix(void) {
    float32_t data[] = {1, 2, 3, 4, 5, 6};
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_create(data, shape, 2, false);

    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(2, t->ndim);
    TEST_ASSERT_EQUAL_UINT64(6, t->size);
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, t->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(3, t->strides[0]);
    TEST_ASSERT_EQUAL_UINT64(1, t->strides[1]);

    tensor_free(t);
}

void test_tensor_create_3d(void) {
    uint64_t shape[] = {2, 2, 2};
    Tensor *t = tensor_zeros(shape, 3, false);

    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(3, t->ndim);
    TEST_ASSERT_EQUAL_UINT64(8, t->size);
    TEST_ASSERT_EQUAL_UINT64(4, t->strides[0]);
    TEST_ASSERT_EQUAL_UINT64(2, t->strides[1]);
    TEST_ASSERT_EQUAL_UINT64(1, t->strides[2]);

    tensor_free(t);
}

void test_tensor_zeros(void) {
    uint64_t shape[] = {2, 2};
    Tensor *t = tensor_zeros(shape, 2, false);

    TEST_ASSERT_NOT_NULL(t);
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, t->data[i]);
    }
    tensor_free(t);
}

void test_tensor_create_requires_grad(void) {
    float32_t val = 1.0f;
    Tensor *t = tensor_create(&val, NULL, 0, true);
    TEST_ASSERT_TRUE(t->requires_grad);
    tensor_free(t);
}

void test_tensor_free_null(void) { tensor_free(NULL); }

void test_tensor_empty_shape(void) {
    uint64_t shape[] = {2, 0, 3};
    Tensor *t = tensor_zeros(shape, 3, false);

    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(0, t->size);
    TEST_ASSERT_NULL(t->data);
    TEST_ASSERT_EQUAL_UINT64(3, t->ndim);

    tensor_free(t);
}

void test_tensor_single_element_vector(void) {
    uint64_t shape[] = {1};
    float32_t val = 5.0f;
    Tensor *t = tensor_create(&val, shape, 1, false);

    TEST_ASSERT_EQUAL_UINT64(1, t->size);
    TEST_ASSERT_EQUAL_UINT64(1, t->ndim);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, t->data[0]);

    tensor_free(t);
}

void test_tensor_squeezable_dims(void) {
    uint64_t shape[] = {1, 5, 1};
    Tensor *t = tensor_zeros(shape, 3, false);

    TEST_ASSERT_EQUAL_UINT64(5, t->size);
    TEST_ASSERT_EQUAL_UINT64(5, t->strides[0]);
    TEST_ASSERT_EQUAL_UINT64(1, t->strides[1]);
    TEST_ASSERT_EQUAL_UINT64(1, t->strides[2]);

    tensor_free(t);
}

void test_linear_to_multidim_1d(void) {
    uint64_t shape[] = {5};
    uint64_t out[1];

    linear_to_multidim_mut(3, shape, 1, out);
    TEST_ASSERT_EQUAL_UINT64(3, out[0]);
}

void test_linear_to_multidim_2d(void) {
    uint64_t shape[] = {2, 3};
    uint64_t out[2];

    linear_to_multidim_mut(4, shape, 2, out);
    TEST_ASSERT_EQUAL_UINT64(1, out[0]);
    TEST_ASSERT_EQUAL_UINT64(1, out[1]);
}

void test_linear_to_multidim_3d(void) {
    uint64_t shape[] = {2, 2, 2};
    uint64_t out[3];

    linear_to_multidim_mut(7, shape, 3, out);
    TEST_ASSERT_EQUAL_UINT64(1, out[0]);
    TEST_ASSERT_EQUAL_UINT64(1, out[1]);
    TEST_ASSERT_EQUAL_UINT64(1, out[2]);

    linear_to_multidim_mut(3, shape, 3, out);
    TEST_ASSERT_EQUAL_UINT64(0, out[0]);
    TEST_ASSERT_EQUAL_UINT64(1, out[1]);
    TEST_ASSERT_EQUAL_UINT64(1, out[2]);
}

void test_multidim_to_linear_1d(void) {
    uint64_t shape[] = {5};
    uint64_t strides[] = {1};
    uint64_t target[] = {3};

    uint64_t off = multidim_to_linear(target, 1, shape, 1, strides);
    TEST_ASSERT_EQUAL_UINT64(3, off);
}

void test_multidim_to_linear_2d(void) {
    uint64_t shape[] = {2, 3};
    uint64_t strides[] = {3, 1};
    uint64_t target[] = {1, 2};

    uint64_t off = multidim_to_linear(target, 2, shape, 2, strides);
    TEST_ASSERT_EQUAL_UINT64(5, off);
}

void test_multidim_to_linear_broadcast_simple(void) {
    uint64_t shape[] = {1, 3};
    uint64_t strides[] = {3, 1};

    uint64_t target[] = {1, 2};

    uint64_t off = multidim_to_linear(target, 2, shape, 2, strides);
    TEST_ASSERT_EQUAL_UINT64(2, off);
}

void test_multidim_to_linear_broadcast_rank_expansion(void) {
    uint64_t shape[] = {3};
    uint64_t strides[] = {1};

    uint64_t target[] = {1, 2};

    uint64_t off = multidim_to_linear(target, 2, shape, 1, strides);
    TEST_ASSERT_EQUAL_UINT64(2, off);
}

void test_multidim_to_linear_broadcast_scalar(void) {
    uint64_t *shape = NULL;
    uint64_t *strides = NULL;

    uint64_t target[] = {1, 1};

    uint64_t off = multidim_to_linear(target, 2, shape, 0, strides);
    TEST_ASSERT_EQUAL_UINT64(0, off);
}

void test_tensor_get_simple_2d(void) {
    float32_t data[] = {1, 2, 3, 4, 5, 6};
    uint64_t shape[] = {2, 3};
    Tensor *t = tensor_create(data, shape, 2, false);

    uint64_t idx[] = {1, 0};
    Tensor *val = tensor_get(t, idx);

    TEST_ASSERT_NOT_NULL(val);
    TEST_ASSERT_EQUAL_UINT64(0, val->ndim);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, val->data[0]);

    tensor_free(val);
    tensor_free(t);
}

void test_tensor_get_scalar(void) {
    float32_t v = 99.0f;
    Tensor *t = tensor_create(&v, NULL, 0, false);

    uint64_t dummy[] = {0};
    Tensor *val = tensor_get(t, dummy);

    TEST_ASSERT_EQUAL_FLOAT(99.0f, val->data[0]);

    tensor_free(val);
    tensor_free(t);
}

void test_tensor_get_broadcast(void) {
    uint64_t shape[] = {2};
    float32_t data[] = {10, 20};
    Tensor *t = tensor_create(data, shape, 1, false);

    uint64_t idx[] = {1};
    Tensor *val = tensor_get(t, idx);
    TEST_ASSERT_EQUAL_FLOAT(20.0f, val->data[0]);

    tensor_free(val);
    tensor_free(t);
}

void test_tensor_data_alignment(void) {
    uint64_t shape[] = {100};
    Tensor *t = tensor_zeros(shape, 1, false);

    TEST_ASSERT_EQUAL_UINT64(0, (uintptr_t)t->data % 64);

    tensor_free(t);
}

void test_tensor_large_dims_strides(void) {
    uint64_t shape[] = {10, 10, 10, 10};
    Tensor *t = tensor_zeros(shape, 4, false);

    TEST_ASSERT_EQUAL_UINT64(1000, t->strides[0]);
    TEST_ASSERT_EQUAL_UINT64(100, t->strides[1]);
    TEST_ASSERT_EQUAL_UINT64(10, t->strides[2]);
    TEST_ASSERT_EQUAL_UINT64(1, t->strides[3]);

    tensor_free(t);
}

void test_tensor_create_rank_limit(void) {
    uint64_t shape[32];
    for (int i = 0; i < 32; i++)
        shape[i] = 1;

    Tensor *t = tensor_zeros(shape, 32, false);
    TEST_ASSERT_NOT_NULL(t);
    TEST_ASSERT_EQUAL_UINT64(32, t->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, t->size);
    tensor_free(t);
}

void test_tensor_print_null(void) { tensor_print(NULL); }

void test_tensor_print_scalar(void) {
    float32_t v = 123.456f;
    Tensor *t = tensor_create(&v, NULL, 0, false);
    tensor_print(t);
    tensor_free(t);
}

void test_tensor_print_vector(void) {
    uint64_t shape[] = {3};
    Tensor *t = tensor_zeros(shape, 1, false);
    tensor_print(t);
    tensor_free(t);
}

void test_tensor_identity_strides(void) {
    uint64_t shape[] = {2, 2};
    Tensor *t = tensor_zeros(shape, 2, false);
    TEST_ASSERT_EQUAL_UINT64(2, t->strides[0]);
    TEST_ASSERT_EQUAL_UINT64(1, t->strides[1]);
    tensor_free(t);
}

void test_tensor_create_shape_copy(void) {
    uint64_t shape[] = {2, 2};
    Tensor *t = tensor_zeros(shape, 2, false);
    shape[0] = 5;
    TEST_ASSERT_EQUAL_UINT64(2, t->shape[0]);
    tensor_free(t);
}

void test_tensor_multidim_to_linear_offset(void) {
    uint64_t shape[] = {3, 3};
    uint64_t strides[] = {3, 1};
    uint64_t target[] = {2, 2};
    TEST_ASSERT_EQUAL_UINT64(8, multidim_to_linear(target, 2, shape, 2, strides));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_tensor_create_scalar);
    RUN_TEST(test_tensor_create_scalar_default_zero);
    RUN_TEST(test_tensor_create_vector);
    RUN_TEST(test_tensor_create_matrix);
    RUN_TEST(test_tensor_create_3d);
    RUN_TEST(test_tensor_zeros);
    RUN_TEST(test_tensor_create_requires_grad);
    RUN_TEST(test_tensor_free_null);
    RUN_TEST(test_tensor_empty_shape);
    RUN_TEST(test_tensor_single_element_vector);
    RUN_TEST(test_tensor_squeezable_dims);
    RUN_TEST(test_tensor_create_rank_limit);
    RUN_TEST(test_tensor_create_shape_copy);
    RUN_TEST(test_linear_to_multidim_1d);
    RUN_TEST(test_linear_to_multidim_2d);
    RUN_TEST(test_linear_to_multidim_3d);
    RUN_TEST(test_multidim_to_linear_1d);
    RUN_TEST(test_multidim_to_linear_2d);
    RUN_TEST(test_multidim_to_linear_broadcast_simple);
    RUN_TEST(test_multidim_to_linear_broadcast_rank_expansion);
    RUN_TEST(test_multidim_to_linear_broadcast_scalar);
    RUN_TEST(test_tensor_multidim_to_linear_offset);
    RUN_TEST(test_tensor_get_simple_2d);
    RUN_TEST(test_tensor_get_scalar);
    RUN_TEST(test_tensor_get_broadcast);
    RUN_TEST(test_tensor_data_alignment);
    RUN_TEST(test_tensor_large_dims_strides);
    RUN_TEST(test_tensor_print_null);
    RUN_TEST(test_tensor_print_scalar);
    RUN_TEST(test_tensor_print_vector);
    RUN_TEST(test_tensor_identity_strides);
    return UNITY_END();
}
