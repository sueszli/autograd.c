#include "autograd.h"
#include "ops.h"
#include "unity.h"
#include <math.h>

void setUp(void) {}
void tearDown(void) { tape_reset(); }

static Tensor *rand_tensor(int64_t rows, int64_t cols, bool requires_grad) {
    Tensor *t = tensor_new(rows, cols, NULL, requires_grad);
    for (int64_t i = 0; i < rows * cols; i++) {
        t->data[i] = 2.0 * rand() / RAND_MAX - 1.0;
    }
    return t;
}

/*
 * reduces f(in) to a scalar with fixed random vectors, loss = r1 @ f(in) @ r2,
 * then compares the analytic gradient of every input element to a central difference.
 */
typedef Tensor *(*Fn)(Tensor **in);

static double loss_of(Fn f, Tensor **in, Tensor *r1, Tensor *r2) {
    double loss = matmul(matmul(r1, f(in)), r2)->data[0];
    tape_reset();
    return loss;
}

static void gradcheck(Fn f, Tensor **in, int64_t n_in) {
    Tensor *out = f(in);
    Tensor *r1 = rand_tensor(1, out->rows, false);
    Tensor *r2 = rand_tensor(out->cols, 1, false);
    backward(matmul(matmul(r1, out), r2));
    tape_reset();
    const double h = 1e-6;
    for (int64_t t = 0; t < n_in; t++) {
        for (int64_t i = 0; in[t]->requires_grad && i < in[t]->rows * in[t]->cols; i++) {
            double x = in[t]->data[i];
            in[t]->data[i] = x + h;
            double up = loss_of(f, in, r1, r2);
            in[t]->data[i] = x - h;
            double down = loss_of(f, in, r1, r2);
            in[t]->data[i] = x;
            double numeric = (up - down) / (2 * h);
            TEST_ASSERT_DOUBLE_WITHIN(1e-6 * (1 + fabs(numeric)), numeric, in[t]->grad[i]);
        }
    }
    for (int64_t t = 0; t < n_in; t++) {
        tensor_free(in[t]);
    }
    tensor_free(r1);
    tensor_free(r2);
}

//
// forward values
//

void test_matmul_forward(void) {
    Tensor *a = tensor_new(2, 2, (double[]){1, 2, 3, 4}, false);
    Tensor *b = tensor_new(2, 1, (double[]){5, 6}, false);
    Tensor *out = matmul(a, b);
    TEST_ASSERT_EQUAL_DOUBLE(17, out->data[0]);
    TEST_ASSERT_EQUAL_DOUBLE(39, out->data[1]);
    tensor_free(a);
    tensor_free(b);
}

void test_transpose_forward(void) {
    Tensor *a = tensor_new(2, 3, (double[]){1, 2, 3, 4, 5, 6}, false);
    Tensor *out = transpose(a);
    TEST_ASSERT_EQUAL_INT64(3, out->rows);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((double[]){1, 4, 2, 5, 3, 6}), out->data, 6);
    tensor_free(a);
}

void test_gather_forward(void) {
    Tensor *table = tensor_new(3, 2, (double[]){1, 2, 3, 4, 5, 6}, false);
    Tensor *out = gather(table, (int64_t[]){2, 0, 2}, 3);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((double[]){5, 6, 1, 2, 5, 6}), out->data, 6);
    tensor_free(table);
}

//
// gradients
//

static Tensor *f_add(Tensor **in) { return add(in[0], in[1]); }
static Tensor *f_add_self(Tensor **in) { return add(in[0], in[0]); }
static Tensor *f_matmul(Tensor **in) { return matmul(in[0], in[1]); }
static Tensor *f_transpose(Tensor **in) { return transpose(in[0]); }
static Tensor *f_gather(Tensor **in) { return gather(in[0], (int64_t[]){1, 0, 1, 3}, 4); }
static Tensor *f_chain(Tensor **in) { return matmul(transpose(matmul(in[0], in[1])), in[2]); }

void test_add_grad(void) { gradcheck(f_add, (Tensor *[]){rand_tensor(3, 4, true), rand_tensor(3, 4, true)}, 2); }
void test_add_self_grad(void) { gradcheck(f_add_self, (Tensor *[]){rand_tensor(3, 4, true)}, 1); }
void test_matmul_grad(void) { gradcheck(f_matmul, (Tensor *[]){rand_tensor(3, 4, true), rand_tensor(4, 5, true)}, 2); }
void test_matmul_frozen_grad(void) { gradcheck(f_matmul, (Tensor *[]){rand_tensor(3, 4, false), rand_tensor(4, 5, true)}, 2); }
void test_transpose_grad(void) { gradcheck(f_transpose, (Tensor *[]){rand_tensor(3, 4, true)}, 1); }
void test_gather_grad(void) { gradcheck(f_gather, (Tensor *[]){rand_tensor(5, 3, true)}, 1); }
void test_chain_grad(void) { gradcheck(f_chain, (Tensor *[]){rand_tensor(3, 4, true), rand_tensor(4, 2, true), rand_tensor(3, 5, true)}, 3); }

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_matmul_forward);
    RUN_TEST(test_transpose_forward);
    RUN_TEST(test_gather_forward);
    RUN_TEST(test_add_grad);
    RUN_TEST(test_add_self_grad);
    RUN_TEST(test_matmul_grad);
    RUN_TEST(test_matmul_frozen_grad);
    RUN_TEST(test_transpose_grad);
    RUN_TEST(test_gather_grad);
    RUN_TEST(test_chain_grad);
    return UNITY_END();
}
