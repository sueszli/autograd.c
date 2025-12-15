#include "ops/arithmetic.h"
#include "tensor.h"
#include "unity.h"
#include <float.h>

void setUp(void) {}
void tearDown(void) {}

static Tensor *create_tensor_1d(float32_t *data, uint64_t size) {
    uint64_t shape[] = {size};
    return tensor_create(data, shape, 1, false);
}

static Tensor *create_scalar(float32_t val) { return tensor_create(&val, NULL, 0, false); }

void test_tensor_add(void) {
    float32_t a_data[] = {1.0f, 2.0f, 3.0f};
    float32_t b_data[] = {4.0f, 5.0f, 6.0f};
    Tensor *a = create_tensor_1d(a_data, 3);
    Tensor *b = create_tensor_1d(b_data, 3);
    Tensor *c = tensor_add(a, b);

    TEST_ASSERT_EQUAL_UINT64(1, c->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, c->shape[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 5.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 7.0f, c->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 9.0f, c->data[2]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_broadcasting(void) {
    Tensor *a = create_scalar(10.0f);
    float32_t b_data[] = {1.0f, 2.0f, 3.0f};
    Tensor *b = create_tensor_1d(b_data, 3);

    Tensor *c = tensor_add(a, b);

    TEST_ASSERT_EQUAL_UINT64(1, c->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, c->shape[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 11.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 12.0f, c->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 13.0f, c->data[2]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_matmul(void) {
    float32_t a_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    uint64_t a_shape[] = {2, 3};
    Tensor *a = tensor_create(a_data, a_shape, 2, false);

    float32_t b_data[] = {7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f};
    uint64_t b_shape[] = {3, 2};
    Tensor *b = tensor_create(b_data, b_shape, 2, false);

    Tensor *c = tensor_matmul(a, b);

    TEST_ASSERT_EQUAL_UINT64(2, c->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, c->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, c->shape[1]);

    TEST_ASSERT_FLOAT_WITHIN(1e-6, 31.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 19.0f, c->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 85.0f, c->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 55.0f, c->data[3]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_sub(void) {
    float32_t a_data[] = {5.0f, 6.0f, 7.0f};
    float32_t b_data[] = {1.0f, 2.0f, 3.0f};
    Tensor *a = create_tensor_1d(a_data, 3);
    Tensor *b = create_tensor_1d(b_data, 3);
    Tensor *c = tensor_sub(a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 4.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 4.0f, c->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 4.0f, c->data[2]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_mul(void) {
    float32_t a_data[] = {2.0f, 3.0f, 4.0f};
    float32_t b_data[] = {2.0f, 3.0f, 4.0f};
    Tensor *a = create_tensor_1d(a_data, 3);
    Tensor *b = create_tensor_1d(b_data, 3);
    Tensor *c = tensor_mul(a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 4.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 9.0f, c->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 16.0f, c->data[2]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_div(void) {
    float32_t a_data[] = {4.0f, 9.0f, 16.0f};
    float32_t b_data[] = {2.0f, 3.0f, 4.0f};
    Tensor *a = create_tensor_1d(a_data, 3);
    Tensor *b = create_tensor_1d(b_data, 3);
    Tensor *c = tensor_div(a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 2.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 3.0f, c->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 4.0f, c->data[2]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_add_broadcast_scalar_lhs(void) {
    Tensor *a = create_scalar(10.0f);
    float32_t b_data[] = {1.0f, 2.0f, 3.0f};
    Tensor *b = create_tensor_1d(b_data, 3);
    Tensor *c = tensor_add(a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 11.0f, c->data[0]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_add_broadcast_scalar_rhs(void) {
    float32_t a_data[] = {1.0f, 2.0f, 3.0f};
    Tensor *a = create_tensor_1d(a_data, 3);
    Tensor *b = create_scalar(10.0f);
    Tensor *c = tensor_add(a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 11.0f, c->data[0]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_sub_broadcast_scalar(void) {
    float32_t a_data[] = {10.0f, 11.0f};
    Tensor *a = create_tensor_1d(a_data, 2);
    Tensor *b = create_scalar(5.0f);
    Tensor *c = tensor_sub(a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 5.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 6.0f, c->data[1]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_mul_broadcast_scalar(void) {
    float32_t a_data[] = {2.0f, 3.0f};
    Tensor *a = create_tensor_1d(a_data, 2);
    Tensor *b = create_scalar(2.0f);
    Tensor *c = tensor_mul(a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 4.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 6.0f, c->data[1]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_div_broadcast_scalar(void) {
    float32_t a_data[] = {10.0f, 20.0f};
    Tensor *a = create_tensor_1d(a_data, 2);
    Tensor *b = create_scalar(2.0f);
    Tensor *c = tensor_div(a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 5.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 10.0f, c->data[1]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_add_zero(void) {
    float32_t a_data[] = {1.0f, 2.0f};
    Tensor *a = create_tensor_1d(a_data, 2);
    Tensor *b = create_scalar(0.0f);
    Tensor *c = tensor_add(a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 2.0f, c->data[1]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_sub_zero(void) {
    float32_t a_data[] = {1.0f, 2.0f};
    Tensor *a = create_tensor_1d(a_data, 2);
    Tensor *b = create_scalar(0.0f);
    Tensor *c = tensor_sub(a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 2.0f, c->data[1]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_mul_zero(void) {
    float32_t a_data[] = {1.0f, 2.0f};
    Tensor *a = create_tensor_1d(a_data, 2);
    Tensor *b = create_scalar(0.0f);
    Tensor *c = tensor_mul(a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, c->data[1]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_mul_one(void) {
    float32_t a_data[] = {1.0f, 2.0f};
    Tensor *a = create_tensor_1d(a_data, 2);
    Tensor *b = create_scalar(1.0f);
    Tensor *c = tensor_mul(a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 2.0f, c->data[1]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_mul_neg_one(void) {
    float32_t a_data[] = {1.0f, 2.0f};
    Tensor *a = create_tensor_1d(a_data, 2);
    Tensor *b = create_scalar(-1.0f);
    Tensor *c = tensor_mul(a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, -1.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, -2.0f, c->data[1]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_div_one(void) {
    float32_t a_data[] = {1.0f, 2.0f};
    Tensor *a = create_tensor_1d(a_data, 2);
    Tensor *b = create_scalar(1.0f);
    Tensor *c = tensor_div(a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 2.0f, c->data[1]);
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_matmul_identity(void) {
    float32_t a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint64_t a_shape[] = {2, 2};
    Tensor *a = tensor_create(a_data, a_shape, 2, false);

    float32_t b_data[] = {1.0f, 0.0f, 0.0f, 1.0f};
    uint64_t b_shape[] = {2, 2};
    Tensor *b = tensor_create(b_data, b_shape, 2, false);

    Tensor *c = tensor_matmul(a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 1.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 2.0f, c->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 3.0f, c->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 4.0f, c->data[3]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

void test_tensor_matmul_zero(void) {
    float32_t a_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint64_t a_shape[] = {2, 2};
    Tensor *a = tensor_create(a_data, a_shape, 2, false);

    float32_t b_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint64_t b_shape[] = {2, 2};
    Tensor *b = tensor_create(b_data, b_shape, 2, false);

    Tensor *c = tensor_matmul(a, b);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, c->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, c->data[1]);

    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_tensor_add);
    RUN_TEST(test_tensor_broadcasting);
    RUN_TEST(test_tensor_matmul);
    RUN_TEST(test_tensor_sub);
    RUN_TEST(test_tensor_mul);
    RUN_TEST(test_tensor_div);
    RUN_TEST(test_tensor_add_broadcast_scalar_lhs);
    RUN_TEST(test_tensor_add_broadcast_scalar_rhs);
    RUN_TEST(test_tensor_sub_broadcast_scalar);
    RUN_TEST(test_tensor_mul_broadcast_scalar);
    RUN_TEST(test_tensor_div_broadcast_scalar);
    RUN_TEST(test_tensor_add_zero);
    RUN_TEST(test_tensor_sub_zero);
    RUN_TEST(test_tensor_mul_zero);
    RUN_TEST(test_tensor_mul_one);
    RUN_TEST(test_tensor_mul_neg_one);
    RUN_TEST(test_tensor_div_one);
    RUN_TEST(test_tensor_matmul_identity);
    RUN_TEST(test_tensor_matmul_zero);
    return UNITY_END();
}
