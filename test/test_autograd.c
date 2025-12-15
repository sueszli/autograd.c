#include "../src/activations.h"
#include "../src/autograd.h"
#include "../src/tensor.h"
#include "unity.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

// Helper to create scalar
static Tensor *scalar(float val, bool requires_grad) {
    uint64_t shape[] = {1};
    Tensor *t = tensor_create(NULL, shape, 0, requires_grad);
    t->data[0] = val;
    return t;
}

// ----------------------------------------------------------------------------
// Basic Arithmetic Tests
// ----------------------------------------------------------------------------

void test_add_simple(void) {
    Tensor *a = scalar(2.0f, true);
    Tensor *b = scalar(3.0f, true);
    Tensor *c = tensor_add(a, b);

    backward(c, NULL);

    // z = a + b => dz/da = 1, dz/db = 1
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

    // z = a - b => dz/da = 1, dz/db = -1
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

    // z = a * b => dz/da = b = 3, dz/db = a = 2
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

    // z = a / b => dz/da = 1/b = 1/3, dz/db = -a/b^2 = -6/9 = -2/3
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 1.0f / 3.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, -2.0f / 3.0f, b->grad->data[0]);

    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

// ----------------------------------------------------------------------------
// Broadcasting Tests
// ----------------------------------------------------------------------------

void test_broadcast_add_vector_scalar(void) {
    // a: [2, 3]
    uint64_t shape_a[] = {2, 3};
    Tensor *a = tensor_create(NULL, shape_a, 2, true);
    for (int i = 0; i < 6; i++)
        a->data[i] = (float)i;

    // b: [1] (scalar-like)
    Tensor *b = scalar(10.0f, true);

    Tensor *c = tensor_add(a, b); // c = a + b (b broadcasts)

    // Backward with ones
    uint64_t shape_c[] = {2, 3};
    Tensor *grad = tensor_create(NULL, shape_c, 2, false);
    for (int i = 0; i < 6; i++)
        grad->data[i] = 1.0f;

    backward(c, grad);

    // dz/da = 1 * grad = ones
    for (int i = 0; i < 6; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-4, 1.0f, a->grad->data[i]);
    }

    // dz/db = sum(grad) = 6
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 6.0f, b->grad->data[0]);

    tensor_free(grad);
    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

void test_broadcast_mul_col_vector(void) {
    // a: [2, 3]
    uint64_t shape_a[] = {2, 3};
    Tensor *a = tensor_create(NULL, shape_a, 2, true);
    // [[1, 2, 3],
    //  [4, 5, 6]]
    for (int i = 0; i < 6; i++)
        a->data[i] = (float)(i + 1);

    // b: [2, 1]
    uint64_t shape_b[] = {2, 1};
    Tensor *b = tensor_create(NULL, shape_b, 2, true);
    // [[2],
    //  [3]]
    b->data[0] = 2.0f;
    b->data[1] = 3.0f;

    Tensor *c = tensor_mul(a, b);
    // c: [[2, 4, 6],
    //     [12, 15, 18]]

    // Let's manually provide grad = ones
    Tensor *grad = tensor_create(NULL, shape_a, 2, false);
    for (int i = 0; i < 6; i++)
        grad->data[i] = 1.0f;

    backward(c, grad);

    // dz/da = grad * b (broadcasted)
    // grad=[1..], b=[[2],[3]]
    // da row 0: 1*2 = 2
    // da row 1: 1*3 = 3
    for (int i = 0; i < 3; i++)
        TEST_ASSERT_FLOAT_WITHIN(1e-4, 2.0f, a->grad->data[i]);
    for (int i = 3; i < 6; i++)
        TEST_ASSERT_FLOAT_WITHIN(1e-4, 3.0f, a->grad->data[i]);

    // dz/db = sum(grad * a, axis=1)
    // row 0: sum([1, 2, 3] * 1) = 6
    // row 1: sum([4, 5, 6] * 1) = 15
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 6.0f, b->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 15.0f, b->grad->data[1]);

    tensor_free(grad);
    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

// ----------------------------------------------------------------------------
// Complex Graph Tests
// ----------------------------------------------------------------------------

void test_chained_ops(void) {
    // z = (a + b) * c
    Tensor *a = scalar(2.0f, true);
    Tensor *b = scalar(3.0f, true);
    Tensor *c = scalar(4.0f, true);

    Tensor *sum = tensor_add(a, b);
    Tensor *z = tensor_mul(sum, c);

    backward(z, NULL);

    // dz/dc = (a+b) = 5
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 5.0f, c->grad->data[0]);
    // dz/da = c = 4
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 4.0f, a->grad->data[0]);
    // dz/db = c = 4
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 4.0f, b->grad->data[0]);

    tensor_free(z);
    tensor_free(sum);
    tensor_free(c);
    tensor_free(b);
    tensor_free(a);
}

void test_variable_reuse_dag(void) {
    // z = a * a + a
    Tensor *a = scalar(3.0f, true);

    Tensor *sq = tensor_mul(a, a);
    Tensor *z = tensor_add(sq, a);

    backward(z, NULL);

    // dz/da = d/da(a^2 + a) = 2a + 1 = 7
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 7.0f, a->grad->data[0]);

    tensor_free(z);
    tensor_free(sq);
    tensor_free(a);
}

// ----------------------------------------------------------------------------
// Activations & Reductions
// ----------------------------------------------------------------------------

void test_relu_grad(void) {
    uint64_t shape[] = {3};
    Tensor *a = tensor_create(NULL, shape, 1, true);
    a->data[0] = 2.0f;
    a->data[1] = -1.0f;
    a->data[2] = 0.0f; // boundary check

    Tensor *y = tensor_relu(a);

    Tensor *grad = tensor_create(NULL, shape, 1, false);
    grad->data[0] = 0.5f;
    grad->data[1] = 0.5f;
    grad->data[2] = 0.5f;

    backward(y, grad);

    // grad is passed where input > 0
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.5f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.0f, a->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.0f, a->grad->data[2]); // strict > 0 usually

    tensor_free(grad);
    tensor_free(y);
    tensor_free(a);
}

void test_softmax_simple(void) {
    uint64_t shape[] = {2};
    Tensor *a = tensor_create(NULL, shape, 1, true);
    a->data[0] = 0.0f;
    a->data[1] = 0.0f; // softmax([0,0]) -> [0.5, 0.5]

    Tensor *p = tensor_softmax(a, 0);

    // Loss = p[0] => pick first class
    // Grad w.r.t input: p - y (where y is one-hot)
    // If we simply backprop grad=[1, 0], we want dp[0]/da

    Tensor *grad = tensor_create(NULL, shape, 1, false);
    grad->data[0] = 1.0f;
    grad->data[1] = 0.0f;

    backward(p, grad);

    // Jacobian of softmax at p=[0.5, 0.5]
    // J = [ p0(1-p0)   -p0p1  ]
    //     [ -p1p0      p1(1-p1) ]
    //   = [ 0.25      -0.25 ]
    //     [ -0.25      0.25 ]
    // grad_in = J^T * grad_out = [0.25, -0.25]^T * [1, 0] = 0.25
    // Actually J is symmetric here
    // result vector: col 0 of J => [0.25, -0.25]

    TEST_ASSERT_FLOAT_WITHIN(1e-4, 0.25f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, -0.25f, a->grad->data[1]);

    tensor_free(grad);
    tensor_free(p);
    tensor_free(a);
}

// ----------------------------------------------------------------------------
// Matrix Multiplication
// ----------------------------------------------------------------------------

void test_matmul_shapes(void) {
    // A: [2, 3], B: [3, 2] -> C: [2, 2]
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

    // dA = grad * B^T
    // grad (2x2) ones. B (3x2). B^T (2x3).
    // result (2x3).
    // B^T = [[1, 3, 5], [2, 4, 6]]
    // grad * B^T = [[1+2, 3+4, 5+6], [1+2, 3+4, 5+6]] = [[3, 7, 11], [3, 7, 11]]

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
    // Call backward twice on same graph
    Tensor *a = scalar(2.0f, true);
    Tensor *z = tensor_mul(a, a); // z = a^2, dz/da = 2a = 4

    backward(z, NULL);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 4.0f, a->grad->data[0]);

    // zero grad? No, PyTorch accumulates. TinyTorch (this impl) replaces?
    // Let's check implementation.
    // accumulate_grad: if (t->grad == NULL) create else add.
    // So it accumulates.

    backward(z, NULL);
    TEST_ASSERT_FLOAT_WITHIN(1e-4, 8.0f, a->grad->data[0]);

    tensor_free(z);
    tensor_free(a);
}

// ----------------------------------------------------------------------------
// Reshape Tests
// ----------------------------------------------------------------------------

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

    // Grad should be reshaped back to {2, 3} and be all ones
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
    return UNITY_END();
}
