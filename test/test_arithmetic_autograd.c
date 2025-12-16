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

void test_matmul_backward_simple(void) {
    uint64_t shape_a[] = {2, 2};
    float32_t data_a[] = {1.0f, 2.0f, 3.0f, 4.0f};

    uint64_t shape_b[] = {2, 2};
    float32_t data_b[] = {0.5f, 0.5f, 0.5f, 0.5f};

    Tensor *a = tensor_create(data_a, shape_a, 2, true);
    Tensor *b = tensor_create(data_b, shape_b, 2, true);

    Tensor *c = tensor_matmul(a, b);
    backward(c);

    TEST_ASSERT_NOT_NULL(a->grad);
    TEST_ASSERT_NOT_NULL(b->grad);

    // dc/da = b^T? For simple sum(c) loss (implicit in backward(c) usually means backward(scalar) or sum if tensor?
    // Wait, backward(tensor) usually sums elements if not scalar.
    // backward() implementation: if tensor is not scalar, it implicitly creates a grad of 1s?
    // Let's check src/autograd.c backward() implementation.
    // But assuming strict mathematical definiton: d(sum(AB))/dA = B^T * 1? No.
    // Let Loss L = sum(C).
    // dL/dA = dL/dC * dC/dA. dL/dC = ones.
    // C_ij = sum_k A_ik B_kj
    // L = sum_ij C_ij
    // dL/dA_xy = sum_ij dC_ij/dA_xy
    // C_ij depends on A_ik where i=x. So C_xj depends on A_xy.
    // C_xj = A_xy * B_yj + ...
    // dC_xj/dA_xy = B_yj.
    // sum_j B_yj.
    // So grad_A should be row sums of B? Or col sums?
    // Let's rely on specific values check.

    // Manual calc:
    // C = [[1.5, 1.5], [3.5, 3.5]]
    // L = 10.0
    // dL/dA = ones @ B.T
    // ones(2,2) @ [[0.5, 0.5], [0.5, 0.5]] = [[1.0, 1.0], [1.0, 1.0]]
    // Wait.
    // dL/dC = [[1, 1], [1, 1]]
    // dL/dA = dL/dC @ B.T = [[1,1],[1,1]] @ [[0.5,0.5],[0.5,0.5]] = [[1,1],[1,1]]
    // dL/dB = A.T @ dL/dC = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]

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
}

void test_matmul_backward_complex(void) {
    // (A + B) @ C
    uint64_t shape[] = {2, 2};
    float32_t val = 1.0f;
    // Tensor *a = tensor_create(&val, shape, 0, true); // scalar broadcasting attempt? No, create scalar 1.0
    // Wait, tensor_create with shape empty is scalar.
    // Let's use 2x2.
    float32_t data[] = {1.0f, 1.0f, 1.0f, 1.0f};
    Tensor *A = tensor_create(data, shape, 2, true);
    Tensor *B = tensor_create(data, shape, 2, true);
    Tensor *C = tensor_create(data, shape, 2, true);

    Tensor *sumAB = tensor_add(A, B);
    Tensor *res = tensor_matmul(sumAB, C);

    backward(res);

    // d(sum(res))/dA = d(sum(res))/d(sumAB) * d(sumAB)/dA
    // d(sum(res))/d(sumAB) = C^T * 1 (approx logic) = [[2,2],[2,2]]
    // d(sumAB)/dA = 1
    // so grad A should be 2 everywhere.

    TEST_ASSERT_NOT_NULL(A->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, A->grad->data[0]);

    tensor_release(A);
    tensor_release(B);
    tensor_release(C);
    tensor_release(sumAB);
    tensor_release(res);
}

void test_add_broadcast_backward(void) {
    uint64_t shape_a[] = {2, 2};
    float32_t data_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint64_t shape_b[] = {2, 1}; // Broadcast over cols
    float32_t data_b[] = {0.5f, 0.5f};

    Tensor *a = tensor_create(data_a, shape_a, 2, true);
    Tensor *b = tensor_create(data_b, shape_b, 2, true);

    Tensor *c = tensor_add(a, b);
    backward(c);

    TEST_ASSERT_NOT_NULL(a->grad);
    TEST_ASSERT_NOT_NULL(b->grad);

    // dL/dC = 1s.
    // dL/dA = 1s.
    // dL/dB = sum_over_cols(1s) = [2, 2]^T?
    // b is (2,1). C is (2,2).
    // b[0,0] adds to C[0,0] and C[0,1].
    // so grad w.r.t b[0,0] is 1+1=2.

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, b->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, b->grad->data[1]);

    tensor_release(a);
    tensor_release(b);
    tensor_release(c);
}

void test_mul_broadcast_backward(void) {
    uint64_t shape_a[] = {2};
    float32_t data_a[] = {2.0f, 3.0f};
    uint64_t shape_b[] = {1}; // Scalar-ish (1,)
    float32_t data_b[] = {4.0f};

    Tensor *a = tensor_create(data_a, shape_a, 1, true);
    Tensor *b = tensor_create(data_b, shape_b, 1, true);

    Tensor *c = tensor_mul(a, b);
    backward(c);

    // c = [8, 12]
    // dL/dC = [1, 1]
    // dL/dA = dL/dC * B = [4, 4]
    // dL/dB = sum(dL/dC * A) = 2*1 + 3*1 = 5

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 4.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 4.0f, a->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 5.0f, b->grad->data[0]);

    tensor_release(a);
    tensor_release(b);
    tensor_release(c);
}

void test_sub_broadcast_backward(void) {
    uint64_t shape_a[] = {2};
    float32_t data_a[] = {5.0f, 6.0f};
    uint64_t shape_b[] = {1};
    float32_t data_b[] = {1.0f};

    Tensor *a = tensor_create(data_a, shape_a, 1, true);
    Tensor *b = tensor_create(data_b, shape_b, 1, true);

    Tensor *c = tensor_sub(a, b);
    backward(c);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -2.0f, b->grad->data[0]); // -1 + -1 = -2

    tensor_release(a);
    tensor_release(b);
    tensor_release(c);
}

void test_div_broadcast_backward(void) {
    uint64_t shape_a[] = {2};
    float32_t data_a[] = {10.0f, 20.0f};
    uint64_t shape_b[] = {1};
    float32_t data_b[] = {2.0f};

    Tensor *a = tensor_create(data_a, shape_a, 1, true);
    Tensor *b = tensor_create(data_b, shape_b, 1, true);

    Tensor *c = tensor_div(a, b);
    backward(c);

    // c = [5, 10]
    // dL/dA = 1/b = 0.5
    // dL/dB = -a/b^2 -> sum()
    // -10/4 = -2.5
    // -20/4 = -5.0
    // sum = -7.5

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.5f, a->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.5f, a->grad->data[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -7.5f, b->grad->data[0]);

    tensor_release(a);
    tensor_release(b);
    tensor_release(c);
}

void test_arithmetic_square(void) {
    uint64_t shape[] = {1};
    float32_t data[] = {3.0f};

    Tensor *a = tensor_create(data, shape, 1, true);
    Tensor *sq = tensor_mul(a, a);
    backward(sq);

    // y = x^2, dy/dx = 2x = 6
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 6.0f, a->grad->data[0]);

    tensor_release(a);
    tensor_release(sq);
}

void test_arithmetic_cubed(void) {
    uint64_t shape[] = {1};
    float32_t data[] = {2.0f};

    Tensor *a = tensor_create(data, shape, 1, true);
    Tensor *sq = tensor_mul(a, a);
    Tensor *cb = tensor_mul(sq, a);
    backward(cb);

    // y = x^3, dy/dx = 3x^2 = 3*4 = 12
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 12.0f, a->grad->data[0]);

    tensor_release(a);
    tensor_release(sq);
    tensor_release(cb);
}

void test_neg_via_mul(void) {
    uint64_t shape[] = {1};
    float32_t data[] = {5.0f};
    float32_t neg[] = {-1.0f};

    Tensor *a = tensor_create(data, shape, 1, true);
    Tensor *minus_one = tensor_create(neg, shape, 1, false);

    Tensor *n = tensor_mul(a, minus_one);
    backward(n);

    TEST_ASSERT_FLOAT_WITHIN(1e-6f, -1.0f, a->grad->data[0]);

    tensor_release(a);
    tensor_release(minus_one);
    tensor_release(n);
}

void test_complex_expression(void) {
    // (a + b) * (a - b) = a^2 - b^2
    uint64_t shape[] = {1};
    float32_t data_a[] = {3.0f};
    float32_t data_b[] = {2.0f};

    Tensor *a = tensor_create(data_a, shape, 1, true);
    Tensor *b = tensor_create(data_b, shape, 1, true);

    Tensor *sum = tensor_add(a, b);
    Tensor *diff = tensor_sub(a, b);
    Tensor *res = tensor_mul(sum, diff);

    backward(res);

    // dA = 2a = 6
    // dB = -2b = -4

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
    RUN_TEST(test_matmul_backward_complex);
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
