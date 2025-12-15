#include "../src/activations.h"
#include "../src/autograd.h"
#include "../src/losses.h"
#include "../src/tensor.h"
#include "unity.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

static Tensor *scalar(float val, bool requires_grad) {
    uint64_t shape[] = {1};
    Tensor *t = tensor_create(NULL, shape, 0, requires_grad);
    t->data[0] = val;
    return t;
}

void test_add_simple(void) {
    Tensor *a = scalar(2.0f, true);
    Tensor *b = scalar(3.0f, true);
    Tensor *c = tensor_add(a, b);

    backward(c, NULL);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 1.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 1.0f, b->grad->data[0]);

    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

void test_sub_simple(void) {
    Tensor *a = scalar(5.0f, true);
    Tensor *b = scalar(3.0f, true);
    Tensor *c = tensor_sub(a, b);

    backward(c, NULL);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 1.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, -1.0f, b->grad->data[0]);

    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

void test_mul_simple(void) {
    Tensor *a = scalar(2.0f, true);
    Tensor *b = scalar(3.0f, true);
    Tensor *c = tensor_mul(a, b);

    backward(c, NULL);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 3.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 2.0f, b->grad->data[0]);

    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

void test_div_simple(void) {
    Tensor *a = scalar(6.0f, true);
    Tensor *b = scalar(3.0f, true);
    Tensor *c = tensor_div(a, b);

    backward(c, NULL);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 1.0f / 3.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, -2.0f / 3.0f, b->grad->data[0]);

    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

void test_broadcast_add_vector_scalar(void) {
    uint64_t shape_a[] = {2, 3};
    Tensor *a = tensor_create(NULL, shape_a, 2, true);
    for (int i = 0; i < 6; i++)
        a->data[i] = (float)i;

    Tensor *b = scalar(10.0f, true);

    Tensor *c = tensor_add(a, b);

    uint64_t shape_c[] = {2, 3};
    Tensor *grad = tensor_create(NULL, shape_c, 2, false);
    for (int i = 0; i < 6; i++)
        grad->data[i] = 1.0f;

    backward(c, grad);

    for (int i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4, 1.0f, a->grad->data[i]);
    }

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 6.0f, b->grad->data[0]);

    tensor_free(grad);
    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

void test_broadcast_mul_col_vector(void) {
    uint64_t shape_a[] = {2, 3};
    Tensor *a = tensor_create(NULL, shape_a, 2, true);

    for (int i = 0; i < 6; i++)
        a->data[i] = (float)(i + 1);

    uint64_t shape_b[] = {2, 1};
    Tensor *b = tensor_create(NULL, shape_b, 2, true);

    b->data[0] = 2.0f;
    b->data[1] = 3.0f;

    Tensor *c = tensor_mul(a, b);

    Tensor *grad = tensor_create(NULL, shape_a, 2, false);
    for (int i = 0; i < 6; i++)
        grad->data[i] = 1.0f;

    backward(c, grad);

    for (int i = 0; i < 3; i++)
        TEST_ASSERT_FLOAT_WITHIN(1e-4, 2.0f, a->grad->data[i]);
    for (int i = 3; i < 6; i++)
        TEST_ASSERT_FLOAT_WITHIN(1e-4, 3.0f, a->grad->data[i]);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 6.0f, b->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 15.0f, b->grad->data[1]);

    tensor_free(grad);
    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

void test_chained_ops(void) {
    Tensor *a = scalar(2.0f, true);
    Tensor *b = scalar(3.0f, true);
    Tensor *c = scalar(4.0f, true);

    Tensor *sum = tensor_add(a, b);
    Tensor *z = tensor_mul(sum, c);

    backward(z, NULL);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 5.0f, c->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 4.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 4.0f, b->grad->data[0]);

    tensor_free(z);
    tensor_free(sum);
    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

void test_variable_reuse_dag(void) {
    Tensor *a = scalar(3.0f, true);

    Tensor *sq = tensor_mul(a, a);
    Tensor *z = tensor_add(sq, a);

    backward(z, NULL);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 7.0f, a->grad->data[0]);

    tensor_free(z);
    tensor_free(sq);
    tensor_free(a);
}

void test_relu_grad(void) {
    uint64_t shape[] = {3};
    Tensor *a = tensor_create(NULL, shape, 1, true);
    a->data[0] = 2.0f;
    a->data[1] = -1.0f;
    a->data[2] = 0.0f;

    Tensor *y = tensor_relu(a);

    Tensor *grad = tensor_create(NULL, shape, 1, false);
    grad->data[0] = 0.5f;
    grad->data[1] = 0.5f;
    grad->data[2] = 0.5f;

    backward(y, grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.5f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.0f, a->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.0f, a->grad->data[2]);

    tensor_free(grad);
    tensor_free(y);
    tensor_free(a);
}

void test_softmax_simple(void) {
    uint64_t shape[] = {2};
    Tensor *a = tensor_create(NULL, shape, 1, true);
    a->data[0] = 0.0f;
    a->data[1] = 0.0f;

    Tensor *p = tensor_softmax(a, 0);

    Tensor *grad = tensor_create(NULL, shape, 1, false);
    grad->data[0] = 1.0f;
    grad->data[1] = 0.0f;

    backward(p, grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.25f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, -0.25f, a->grad->data[1]);

    tensor_free(grad);
    tensor_free(p);
    tensor_free(a);
}

void test_matmul_shapes(void) {
    uint64_t shape_a[] = {2, 3};
    Tensor *a = tensor_create(NULL, shape_a, 2, true);
    for (int i = 0; i < 6; i++)
        a->data[i] = (float)(i + 1);

    uint64_t shape_b[] = {3, 2};
    Tensor *b = tensor_create(NULL, shape_b, 2, true);
    for (int i = 0; i < 6; i++)
        b->data[i] = (float)(i + 1);

    Tensor *c = tensor_matmul(a, b);

    uint64_t shape_c[] = {2, 2};
    Tensor *grad = tensor_create(NULL, shape_c, 2, false);
    for (int i = 0; i < 4; i++)
        grad->data[i] = 1.0f;

    backward(c, grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 3.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 7.0f, a->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 11.0f, a->grad->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 3.0f, a->grad->data[3]);

    tensor_free(grad);
    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

void test_accumulate_grad(void) {
    Tensor *a = scalar(2.0f, true);
    Tensor *z = tensor_mul(a, a);

    backward(z, NULL);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 4.0f, a->grad->data[0]);

    backward(z, NULL);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 8.0f, a->grad->data[0]);

    tensor_free(z);
    tensor_free(a);
}

void test_reshape_simple(void) {
    uint64_t shape[] = {2, 3};
    Tensor *a = tensor_create(NULL, shape, 2, true);
    for (int i = 0; i < 6; i++)
        a->data[i] = (float)(i + 1);

    int64_t new_shape[] = {3, 2};
    Tensor *b = tensor_reshape(a, new_shape, 2);

    Tensor *grad = tensor_create(NULL, (uint64_t[]){3, 2}, 2, false);
    for (int i = 0; i < 6; i++)
        grad->data[i] = 1.0f;

    backward(b, grad);

    TEST_ASSERT_EQUAL_UINT64(2, a->grad->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, a->grad->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, a->grad->shape[1]);
    for (int i = 0; i < 6; i++)
        TEST_ASSERT_FLOAT_WITHIN(1e-4, 1.0f, a->grad->data[i]);

    tensor_free(grad);
    tensor_free(b);
    tensor_free(a);
}

void test_reshape_flatten(void) {
    uint64_t shape[] = {2, 2, 2};
    Tensor *a = tensor_create(NULL, shape, 3, true);
    for (int i = 0; i < 8; i++)
        a->data[i] = (float)i;

    int64_t new_shape[] = {8};
    Tensor *b = tensor_reshape(a, new_shape, 1);

    Tensor *grad = tensor_create(NULL, (uint64_t[]){8}, 1, false);
    for (int i = 0; i < 8; i++)
        grad->data[i] = 0.5f;

    backward(b, grad);

    TEST_ASSERT_EQUAL_UINT64(3, a->grad->ndim);
    for (int i = 0; i < 8; i++)
        TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.5f, a->grad->data[i]);

    tensor_free(grad);
    tensor_free(b);
    tensor_free(a);
}

void test_sum_all(void) {
    uint64_t shape[] = {2, 2};
    Tensor *a = tensor_create(NULL, shape, 2, true);
    for (int i = 0; i < 4; i++)
        a->data[i] = (float)i;

    Tensor *s = tensor_sum(a, 0, false);

    Tensor *grad = tensor_create(NULL, (uint64_t[]){2}, 1, false);
    grad->data[0] = 1.0f;
    grad->data[1] = 2.0f;

    backward(s, grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 1.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 2.0f, a->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 1.0f, a->grad->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 2.0f, a->grad->data[3]);

    tensor_free(grad);
    tensor_free(s);
    tensor_free(a);
}

void test_sum_keepdims(void) {
    uint64_t shape[] = {2, 2};
    Tensor *a = tensor_create(NULL, shape, 2, true);

    for (int i = 0; i < 4; i++)
        a->data[i] = 1.0f;

    Tensor *s = tensor_sum(a, 1, true);

    Tensor *grad = tensor_create(NULL, (uint64_t[]){2, 1}, 2, false);
    grad->data[0] = 5.0f;
    grad->data[1] = 10.0f;

    backward(s, grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 5.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 5.0f, a->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 10.0f, a->grad->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 10.0f, a->grad->data[3]);

    tensor_free(grad);
    tensor_free(s);
    tensor_free(a);
}

void test_sigmoid(void) {
    uint64_t shape[] = {3};
    Tensor *a = tensor_create(NULL, shape, 1, true);
    a->data[0] = 0.0f;
    a->data[1] = 100.0f;
    a->data[2] = -100.0f;

    Tensor *s = tensor_sigmoid(a);

    Tensor *grad = tensor_create(NULL, shape, 1, false);
    grad->data[0] = 1.0f;
    grad->data[1] = 1.0f;
    grad->data[2] = 1.0f;

    backward(s, grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.25f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.0f, a->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.0f, a->grad->data[2]);

    tensor_free(grad);
    tensor_free(s);
    tensor_free(a);
}

void test_relu_negative(void) {
    Tensor *a = scalar(-5.0f, true);
    Tensor *r = tensor_relu(a);
    backward(r, NULL);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.0f, a->grad->data[0]);
    tensor_free(r);
    tensor_free(a);
}

void test_diamond_graph(void) {
    Tensor *a = scalar(2.0f, true);
    Tensor *factor2 = scalar(2.0f, false);
    Tensor *factor3 = scalar(3.0f, false);

    Tensor *b = tensor_mul(a, factor2);
    Tensor *c = tensor_mul(a, factor3);
    Tensor *d = tensor_add(b, c);

    backward(d, NULL);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 5.0f, a->grad->data[0]);

    tensor_free(d);
    tensor_free(c);
    tensor_free(b);
    tensor_free(factor3);
    tensor_free(factor2);
    tensor_free(a);
}

void test_deep_chain(void) {
    Tensor *x = scalar(1.0f, true);
    Tensor *h1 = tensor_mul(x, x);
    Tensor *h2 = tensor_mul(h1, h1);
    Tensor *h3 = tensor_mul(h2, h2);

    backward(h3, NULL);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 8.0f, x->grad->data[0]);

    tensor_free(h3);
    tensor_free(h2);
    tensor_free(h1);
    tensor_free(x);
}

void test_multi_branch_reduction(void) {
    uint64_t shape[] = {2, 2};
    Tensor *a = tensor_create(NULL, shape, 2, true);
    for (int i = 0; i < 4; i++)
        a->data[i] = 2.0f;

    Tensor *s1 = tensor_sum(a, 0, false);
    Tensor *s2 = tensor_sum(s1, 0, false);

    Tensor *z = tensor_mul(s2, s2);

    backward(z, NULL);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4, 16.0f, a->grad->data[i]);
    }

    tensor_free(z);
    tensor_free(s2);
    tensor_free(s1);
    tensor_free(a);
}

void test_disconnected_graph(void) {
    Tensor *a = scalar(1.0f, true);
    Tensor *b = scalar(2.0f, true);
    Tensor *c = tensor_mul(b, b);

    backward(c, NULL);

    TEST_ASSERT_NULL(a->grad);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 4.0f, b->grad->data[0]);

    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

void test_complex_broadcast_add(void) {
    uint64_t shape_a[] = {2, 1, 4};
    Tensor *a = tensor_create(NULL, shape_a, 3, true);

    uint64_t shape_b[] = {4};
    Tensor *b = tensor_create(NULL, shape_b, 1, true);

    Tensor *c = tensor_add(a, b);

    Tensor *grad = tensor_create(NULL, shape_a, 3, false);
    for (int i = 0; i < 8; i++)
        grad->data[i] = 1.0f;

    backward(c, grad);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4, 2.0f, b->grad->data[i]);
    }

    tensor_free(grad);
    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

void test_gelu_grad(void) {
    Tensor *x = scalar(0.0f, true);
    Tensor *y = tensor_gelu(x);
    backward(y, NULL);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.5f, x->grad->data[0]);
    tensor_free(y);
    tensor_free(x);

    x = scalar(1.0f, true);
    y = tensor_gelu(x);
    backward(y, NULL);

    tensor_free(y);
    tensor_free(x);
}

void test_transpose_grad(void) {
    uint64_t shape[] = {2, 3};
    Tensor *a = tensor_create(NULL, shape, 2, true);
    for (int i = 0; i < 6; i++)
        a->data[i] = (float)i;

    Tensor *b = tensor_transpose(a, 0, 1);

    Tensor *grad_b = tensor_create(NULL, (uint64_t[]){3, 2}, 2, false);
    for (int i = 0; i < 6; i++)
        grad_b->data[i] = (float)i;

    backward(b, grad_b);

    float expected[] = {0, 2, 4, 1, 3, 5};
    for (int i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4, expected[i], a->grad->data[i]);
    }

    tensor_free(grad_b);
    tensor_free(b);
    tensor_free(a);
}

void test_getitem_grad(void) {
    uint64_t shape[] = {2, 2};
    Tensor *a = tensor_create(NULL, shape, 2, true);
    a->data[0] = 1;
    a->data[1] = 2;
    a->data[2] = 3;
    a->data[3] = 4;

    uint64_t idx[] = {1, 0};
    Tensor *y = tensor_get(a, idx);

    backward(y, NULL);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.0f, a->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 1.0f, a->grad->data[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.0f, a->grad->data[3]);

    tensor_free(y);
    tensor_free(a);
}

void test_mse_backward(void) {
    Tensor *pred = scalar(2.0f, true);
    Tensor *target = scalar(1.0f, false);
    Tensor *loss = mse_loss(pred, target);

    backward(loss, NULL);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 2.0f, pred->grad->data[0]);

    tensor_free(loss);
    tensor_free(target);
    tensor_free(pred);
}

void test_bce_backward(void) {
    Tensor *pred = scalar(0.8f, true);
    Tensor *target = scalar(1.0f, false);
    Tensor *loss = binary_cross_entropy_loss(pred, target);

    backward(loss, NULL);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, -1.25f, pred->grad->data[0]);

    tensor_free(loss);
    tensor_free(target);
    tensor_free(pred);
}

void test_crossentropy_backward(void) {
    float l_data[] = {2.0f, 1.0f};
    uint64_t shape[] = {1, 2};
    Tensor *logits = tensor_create(l_data, shape, 2, true);

    float t_data[] = {0.0f};
    Tensor *targets = tensor_create(t_data, (uint64_t[]){1}, 1, false);

    Tensor *loss = cross_entropy_loss(logits, targets);

    backward(loss, NULL);

    TEST_ASSERT_FLOAT_WITHIN(1e-4, -0.268941f, logits->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.268941f, logits->grad->data[1]);

    tensor_free(loss);
    tensor_free(targets);
    tensor_free(logits);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_add_simple);
    RUN_TEST(test_sub_simple);
    RUN_TEST(test_mul_simple);
    RUN_TEST(test_div_simple);
    RUN_TEST(test_broadcast_add_vector_scalar);
    RUN_TEST(test_broadcast_mul_col_vector);
    RUN_TEST(test_chained_ops);
    RUN_TEST(test_variable_reuse_dag);
    RUN_TEST(test_accumulate_grad);
    RUN_TEST(test_relu_grad);
    RUN_TEST(test_softmax_simple);
    RUN_TEST(test_matmul_shapes);
    RUN_TEST(test_reshape_simple);
    RUN_TEST(test_reshape_flatten);
    RUN_TEST(test_sum_all);
    RUN_TEST(test_sum_keepdims);
    RUN_TEST(test_sigmoid);
    RUN_TEST(test_relu_negative);
    RUN_TEST(test_diamond_graph);
    RUN_TEST(test_deep_chain);
    RUN_TEST(test_multi_branch_reduction);
    RUN_TEST(test_disconnected_graph);
    RUN_TEST(test_complex_broadcast_add);
    RUN_TEST(test_gelu_grad);
    RUN_TEST(test_transpose_grad);
    RUN_TEST(test_getitem_grad);
    RUN_TEST(test_mse_backward);
    RUN_TEST(test_bce_backward);
    RUN_TEST(test_crossentropy_backward);
    return UNITY_END();
}
