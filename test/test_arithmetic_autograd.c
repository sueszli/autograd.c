#include "autograd.h"
#include "ops/arithmetic.h"
#include "ops/reductions.h"
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

void test_matmul_backward_simple(void) {
    uint64_t shape_a[] = {2, 2};
    float32_t data_a[] = {1.0f, 2.0f, 3.0f, 4.0f};

    uint64_t shape_b[] = {2, 2};
    float32_t data_b[] = {0.5f, 0.5f, 0.5f, 0.5f};

    Tensor *a = tensor_create(data_a, shape_a, 2, true);
    Tensor *b = tensor_create(data_b, shape_b, 2, true);

    Tensor *c = tensor_matmul(a, b);
    Tensor *sum1 = tensor_sum(c, 0, false);
    Tensor *loss = tensor_sum(sum1, 0, false);
    backward(loss);

    TEST_ASSERT_NOT_NULL(a->grad);
    TEST_ASSERT_NOT_NULL(b->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, a->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, a->grad->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, a->grad->data[3]);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 4.0f, b->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 4.0f, b->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 6.0f, b->grad->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 6.0f, b->grad->data[3]);

    tensor_release(a);
    tensor_release(b);
    tensor_release(c);
    tensor_release(sum1);
    tensor_release(loss);
}

void test_add_broadcast_backward(void) {
    uint64_t shape_a[] = {2, 2};
    float32_t data_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint64_t shape_b[] = {2, 1};
    float32_t data_b[] = {0.5f, 0.5f};

    Tensor *a = tensor_create(data_a, shape_a, 2, true);
    Tensor *b = tensor_create(data_b, shape_b, 2, true);

    Tensor *c = tensor_add(a, b);
    Tensor *sum1 = tensor_sum(c, 0, false);
    Tensor *loss = tensor_sum(sum1, 0, false);
    backward(loss);

    TEST_ASSERT_NOT_NULL(a->grad);
    TEST_ASSERT_NOT_NULL(b->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, b->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, b->grad->data[1]);

    tensor_release(a);
    tensor_release(b);
    tensor_release(c);
    tensor_release(sum1);
    tensor_release(loss);
}

void test_mul_broadcast_backward(void) {
    uint64_t shape_a[] = {2};
    float32_t data_a[] = {2.0f, 3.0f};
    uint64_t shape_b[] = {1};
    float32_t data_b[] = {4.0f};

    Tensor *a = tensor_create(data_a, shape_a, 1, true);
    Tensor *b = tensor_create(data_b, shape_b, 1, true);

    Tensor *c = tensor_mul(a, b);
    Tensor *loss = tensor_sum(c, 0, false);
    backward(loss);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 4.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 4.0f, a->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 5.0f, b->grad->data[0]);

    tensor_release(a);
    tensor_release(b);
    tensor_release(c);
    tensor_release(loss);
}

void test_sub_broadcast_backward(void) {
    uint64_t shape_a[] = {2};
    float32_t data_a[] = {5.0f, 6.0f};
    uint64_t shape_b[] = {1};
    float32_t data_b[] = {1.0f};

    Tensor *a = tensor_create(data_a, shape_a, 1, true);
    Tensor *b = tensor_create(data_b, shape_b, 1, true);

    Tensor *c = tensor_sub(a, b);
    Tensor *loss = tensor_sum(c, 0, false);
    backward(loss);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -2.0f, b->grad->data[0]);

    tensor_release(a);
    tensor_release(b);
    tensor_release(c);
    tensor_release(loss);
}

void test_div_broadcast_backward(void) {
    uint64_t shape_a[] = {2};
    float32_t data_a[] = {10.0f, 20.0f};
    uint64_t shape_b[] = {1};
    float32_t data_b[] = {2.0f};

    Tensor *a = tensor_create(data_a, shape_a, 1, true);
    Tensor *b = tensor_create(data_b, shape_b, 1, true);

    Tensor *c = tensor_div(a, b);
    Tensor *loss = tensor_sum(c, 0, false);
    backward(loss);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.5f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.5f, a->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -7.5f, b->grad->data[0]);

    tensor_release(a);
    tensor_release(b);
    tensor_release(c);
    tensor_release(loss);
}

void test_arithmetic_square(void) {
    uint64_t shape[] = {};
    float32_t data[] = {3.0f};

    Tensor *a = tensor_create(data, shape, 0, true);
    Tensor *sq = tensor_mul(a, a);
    backward(sq);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 6.0f, a->grad->data[0]);

    tensor_release(a);
    tensor_release(sq);
}

void test_arithmetic_cubed(void) {
    uint64_t shape[] = {};
    float32_t data[] = {2.0f};

    Tensor *a = tensor_create(data, shape, 0, true);
    Tensor *sq = tensor_mul(a, a);
    Tensor *cb = tensor_mul(sq, a);
    backward(cb);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 12.0f, a->grad->data[0]);

    tensor_release(a);
    tensor_release(sq);
    tensor_release(cb);
}

void test_neg_via_mul(void) {
    uint64_t shape[] = {};
    float32_t data[] = {5.0f};
    float32_t neg[] = {-1.0f};

    Tensor *a = tensor_create(data, shape, 0, true);
    Tensor *minus_one = tensor_create(neg, shape, 0, false);

    Tensor *n = tensor_mul(a, minus_one);
    backward(n);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -1.0f, a->grad->data[0]);

    tensor_release(a);
    tensor_release(minus_one);
    tensor_release(n);
}

void test_complex_expression(void) {

    uint64_t shape[] = {};
    float32_t data_a[] = {3.0f};
    float32_t data_b[] = {2.0f};

    Tensor *a = tensor_create(data_a, shape, 0, true);
    Tensor *b = tensor_create(data_b, shape, 0, true);

    Tensor *sum = tensor_add(a, b);
    Tensor *diff = tensor_sub(a, b);
    Tensor *res = tensor_mul(sum, diff);

    backward(res);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 6.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -4.0f, b->grad->data[0]);

    tensor_release(a);
    tensor_release(b);
    tensor_release(sum);
    tensor_release(diff);
    tensor_release(res);
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
    RUN_TEST(test_matmul_backward_simple);

    RUN_TEST(test_add_broadcast_backward);
    RUN_TEST(test_mul_broadcast_backward);
    RUN_TEST(test_sub_broadcast_backward);
    RUN_TEST(test_div_broadcast_backward);
    RUN_TEST(test_arithmetic_square);
    RUN_TEST(test_arithmetic_cubed);
    RUN_TEST(test_neg_via_mul);
    RUN_TEST(test_complex_expression);
    return UNITY_END();
}
