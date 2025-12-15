#include "autograd.h"
#include "tensor.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EPSILON 1e-4f

void setUp(void) {}
void tearDown(void) {}

static void assert_float_equal(float a, float b) {
    if (fabs(a - b) > EPSILON) {
        printf("assertion failed: %f != %f\n", a, b);
        exit(1);
    }
}

static void test_scalar_add(void) {
    uint64_t shape[] = {1};
    Tensor *a = tensor_create(NULL, shape, 0, true);
    a->data[0] = 2.0f;
    Tensor *b = tensor_create(NULL, shape, 0, true);
    b->data[0] = 3.0f;
    Tensor *c = tensor_add(a, b);

    backward(c, NULL);

    assert_float_equal(a->grad->data[0], 1.0f);
    assert_float_equal(b->grad->data[0], 1.0f);

    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

static void test_scalar_mul(void) {
    uint64_t shape[] = {1};
    Tensor *a = tensor_create(NULL, shape, 0, true);
    a->data[0] = 2.0f;
    Tensor *b = tensor_create(NULL, shape, 0, true);
    b->data[0] = 3.0f;
    Tensor *c = tensor_mul(a, b);

    backward(c, NULL);

    assert_float_equal(a->grad->data[0], 3.0f);
    assert_float_equal(b->grad->data[0], 2.0f);

    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

static void test_expression_1(void) {
    uint64_t shape[] = {1};
    Tensor *a = tensor_create(NULL, shape, 0, true);
    a->data[0] = 2.0f;
    Tensor *b = tensor_create(NULL, shape, 0, true);
    b->data[0] = 3.0f;
    Tensor *c = tensor_create(NULL, shape, 0, true);
    c->data[0] = 4.0f;

    Tensor *ab = tensor_mul(a, b);
    Tensor *z = tensor_add(ab, c);

    backward(z, NULL);

    assert_float_equal(a->grad->data[0], 3.0f);
    assert_float_equal(b->grad->data[0], 2.0f);
    assert_float_equal(c->grad->data[0], 1.0f);

    tensor_free(z);
    tensor_free(ab);
    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

static void test_branching(void) {
    uint64_t shape[] = {1};
    Tensor *a = tensor_create(NULL, shape, 0, true);
    a->data[0] = 3.0f;

    Tensor *b = tensor_mul(a, a);

    backward(b, NULL);

    assert_float_equal(a->grad->data[0], 6.0f);

    tensor_free(b);
    tensor_free(a);
}

static void test_tensor_matmul(void) {
    uint64_t shape_a[] = {2, 3};
    Tensor *a = tensor_create(NULL, shape_a, 2, true);
    for (int i = 0; i < 6; i++)
        a->data[i] = (float)i;

    uint64_t shape_b[] = {3, 2};
    Tensor *b = tensor_create(NULL, shape_b, 2, true);
    for (int i = 0; i < 6; i++)
        b->data[i] = 1.0f;

    Tensor *c = tensor_matmul(a, b);

    uint64_t shape_c[] = {2, 2};
    Tensor *grad_c = tensor_create(NULL, shape_c, 2, false);
    for (int i = 0; i < 4; i++)
        grad_c->data[i] = 1.0f;

    backward(c, grad_c);

    assert_float_equal(a->grad->data[0], 2.0f);
    assert_float_equal(a->grad->data[5], 2.0f);

    tensor_free(grad_c);
    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

static void test_broadcast_add(void) {
    uint64_t shape_a[] = {2, 3};
    Tensor *a = tensor_create(NULL, shape_a, 2, true);
    for (int i = 0; i < 6; i++)
        a->data[i] = 0.0f;

    uint64_t shape_b[] = {3};
    Tensor *b = tensor_create(NULL, shape_b, 1, true);
    for (int i = 0; i < 3; i++)
        b->data[i] = 0.0f;

    Tensor *c = tensor_add(a, b);

    uint64_t shape_c[] = {2, 3};
    Tensor *grad_c = tensor_create(NULL, shape_c, 2, false);
    for (int i = 0; i < 6; i++)
        grad_c->data[i] = 1.0f;

    backward(c, grad_c);

    assert_float_equal(a->grad->data[0], 1.0f);
    assert_float_equal(b->grad->data[0], 2.0f);
    assert_float_equal(b->grad->data[1], 2.0f);
    assert_float_equal(b->grad->data[2], 2.0f);

    tensor_free(grad_c);
    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_scalar_add);
    RUN_TEST(test_scalar_mul);
    RUN_TEST(test_expression_1);
    RUN_TEST(test_branching);
    RUN_TEST(test_tensor_matmul);
    RUN_TEST(test_broadcast_add);
    return UNITY_END();
}
