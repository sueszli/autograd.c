#include "autograd.h"
#include "ops.h"
#include "std/testing.h"
#include "unity.h"
#include <math.h>

void setUp(void) {}
void tearDown(void) {
    graph_reset();
    fixed_buffer_allocator_reset(&perm_mem);
}

static Tensor *rand_tensor(int64_t rows, int64_t cols, bool requires_grad) {
    Tensor *t = tensor_new(rows, cols, NULL, requires_grad);
    random_uniform(t->data, rows * cols, -1.0, 1.0);
    return t;
}

/*
 * reduces f(in) to a scalar with fixed random vectors, loss = r1 @ f(in) @ r2,
 * then compares the analytic gradient of every input element to a central difference.
 */
typedef Tensor *(*Fn)(Tensor **in);

typedef struct {
    Fn f;
    Tensor **in, *r1, *r2;
} Probe;

static double loss_of(void *ctx) {
    const Probe *p = ctx;
    double loss = matmul(matmul(p->r1, p->f(p->in)), p->r2)->data[0];
    graph_reset();
    return loss;
}

static void gradcheck(Fn f, Tensor **in, int64_t n_in) {
    Tensor *out = f(in);
    Tensor *r1 = rand_tensor(1, out->rows, false);
    Tensor *r2 = rand_tensor(out->cols, 1, false);
    // grads start at 0.5, so a backward that overwrites instead of accumulating fails
    for (int64_t t = 0; t < n_in; t++) {
        for (int64_t i = 0; in[t]->requires_grad && i < in[t]->rows * in[t]->cols; i++) {
            in[t]->grad[i] = 0.5;
        }
    }
    backward(matmul(matmul(r1, out), r2));
    graph_reset();
    Probe probe = {f, in, r1, r2};
    for (int64_t t = 0; t < n_in; t++) {
        for (int64_t i = 0; in[t]->requires_grad && i < in[t]->rows * in[t]->cols; i++) {
            double numeric = central_difference(loss_of, &probe, &in[t]->data[i], 1e-6);
            assert_close(in[t]->grad[i] - 0.5, numeric, 1e-6, 1e-6, "grad");
        }
    }
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
}

void test_transpose_forward(void) {
    Tensor *a = tensor_new(2, 3, (double[]){1, 2, 3, 4, 5, 6}, false);
    Tensor *out = transpose(a);
    TEST_ASSERT_EQUAL_INT64(3, out->rows);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((double[]){1, 4, 2, 5, 3, 6}), out->data, 6);
}

void test_gather_forward(void) {
    Tensor *table = tensor_new(3, 2, (double[]){1, 2, 3, 4, 5, 6}, false);
    Tensor *out = gather(table, (int64_t[]){2, 0, 2}, 3);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((double[]){5, 6, 1, 2, 5, 6}), out->data, 6);
}

void test_rmsnorm_forward(void) {
    Tensor *a = tensor_new(1, 2, (double[]){3, 4}, false);
    Tensor *out = rmsnorm(a, 0.0);
    // rms of (3, 4) is sqrt(12.5)
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, 3 / sqrt(12.5), out->data[0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, 4 / sqrt(12.5), out->data[1]);
}

void test_causal_softmax_forward(void) {
    Tensor *a = tensor_new(2, 2, (double[]){1, 5, 2, 2}, false);
    Tensor *out = softmax_rows(causal_mask(a, -1e9));
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((double[]){1, 0, 0.5, 0.5}), out->data, 4);
}

// large logits overflow exp unless each row subtracts its max first
void test_softmax_large_logits(void) {
    Tensor *a = tensor_new(2, 2, (double[]){1000, 0, 0, 1000}, false);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((double[]){1, 0, 0, 1}), softmax_rows(a)->data, 4);
}

void test_relu_scale_forward(void) {
    Tensor *a = tensor_new(1, 3, (double[]){-1, 0, 2}, false);
    Tensor *out = scale(relu(a), 0.5);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((double[]){0, 0, 1}), out->data, 3);
}

void test_slice_concat_forward(void) {
    Tensor *a = tensor_new(2, 3, (double[]){1, 2, 3, 4, 5, 6}, false);
    Tensor *out = concat_cols((Tensor *[]){slice_cols(a, 2, 1), slice_cols(a, 0, 2)}, 2);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((double[]){3, 1, 2, 6, 4, 5}), out->data, 6);
}

void test_cross_entropy_forward(void) {
    // uniform logits give log(cols) per row, the masked row is ignored
    Tensor *logits = tensor_new(2, 4, (double[]){0, 0, 0, 0, 9, 1, 2, 3}, false);
    Tensor *out = masked_cross_entropy(logits, (int64_t[]){1, 2}, (double[]){1, 0});
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, log(4.0), out->data[0]);
    Tensor *none = masked_cross_entropy(logits, (int64_t[]){1, 2}, (double[]){0, 0});
    TEST_ASSERT_EQUAL_DOUBLE(0.0, none->data[0]);
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
static Tensor *f_scale(Tensor **in) { return scale(in[0], -0.3); }
static Tensor *f_relu(Tensor **in) { return relu(in[0]); }
static Tensor *f_rmsnorm(Tensor **in) { return rmsnorm(in[0], 1e-5); }
// a small mask value, -1e9 would make the finite difference lose all precision
static Tensor *f_causal_mask(Tensor **in) { return causal_mask(in[0], -5.0); }
static Tensor *f_softmax_rows(Tensor **in) { return softmax_rows(in[0]); }
static Tensor *f_split_merge(Tensor **in) { return concat_cols((Tensor *[]){slice_cols(in[0], 3, 1), slice_cols(in[0], 0, 3), in[1]}, 3); }
static Tensor *f_cross_entropy(Tensor **in) { return masked_cross_entropy(in[0], (int64_t[]){2, 0, 4}, (double[]){1, 0, 1}); }

// non-leaf inputs to matmul and softmax once lost their gradients
static Tensor *f_mlp(Tensor **in) { return matmul(relu(matmul(in[0], in[1])), in[2]); }
static Tensor *f_softmax_of_matmul(Tensor **in) { return softmax_rows(matmul(in[0], in[1])); }

// one causal attention head: softmax(mask(q @ k^T / sqrt(d))) @ v
static Tensor *f_attention(Tensor **in) {
    Tensor *xn = rmsnorm(in[0], 1e-5);
    Tensor *q = matmul(xn, in[1]), *k = matmul(xn, in[2]), *v = matmul(xn, in[3]);
    return matmul(softmax_rows(causal_mask(scale(matmul(q, transpose(k)), 0.5), -1e9)), v);
}

void test_add_grad(void) { gradcheck(f_add, (Tensor *[]){rand_tensor(3, 4, true), rand_tensor(3, 4, true)}, 2); }
void test_add_self_grad(void) { gradcheck(f_add_self, (Tensor *[]){rand_tensor(3, 4, true)}, 1); }
void test_matmul_grad(void) { gradcheck(f_matmul, (Tensor *[]){rand_tensor(3, 4, true), rand_tensor(4, 5, true)}, 2); }
void test_matmul_frozen_grad(void) { gradcheck(f_matmul, (Tensor *[]){rand_tensor(3, 4, false), rand_tensor(4, 5, true)}, 2); }
void test_transpose_grad(void) { gradcheck(f_transpose, (Tensor *[]){rand_tensor(3, 4, true)}, 1); }
void test_gather_grad(void) { gradcheck(f_gather, (Tensor *[]){rand_tensor(5, 3, true)}, 1); }
void test_chain_grad(void) { gradcheck(f_chain, (Tensor *[]){rand_tensor(3, 4, true), rand_tensor(4, 2, true), rand_tensor(3, 5, true)}, 3); }
void test_scale_grad(void) { gradcheck(f_scale, (Tensor *[]){rand_tensor(3, 4, true)}, 1); }
void test_relu_grad(void) { gradcheck(f_relu, (Tensor *[]){rand_tensor(3, 4, true)}, 1); }
void test_rmsnorm_grad(void) { gradcheck(f_rmsnorm, (Tensor *[]){rand_tensor(3, 4, true)}, 1); }
void test_causal_mask_grad(void) { gradcheck(f_causal_mask, (Tensor *[]){rand_tensor(4, 4, true)}, 1); }
void test_softmax_rows_grad(void) { gradcheck(f_softmax_rows, (Tensor *[]){rand_tensor(3, 4, true)}, 1); }
void test_split_merge_grad(void) { gradcheck(f_split_merge, (Tensor *[]){rand_tensor(3, 4, true), rand_tensor(3, 2, true)}, 2); }
void test_cross_entropy_grad(void) { gradcheck(f_cross_entropy, (Tensor *[]){rand_tensor(3, 5, true)}, 1); }
void test_mlp_grad(void) { gradcheck(f_mlp, (Tensor *[]){rand_tensor(3, 4, false), rand_tensor(4, 8, true), rand_tensor(8, 2, true)}, 3); }
void test_softmax_of_matmul_grad(void) { gradcheck(f_softmax_of_matmul, (Tensor *[]){rand_tensor(3, 4, false), rand_tensor(4, 5, true)}, 2); }
void test_attention_grad(void) { gradcheck(f_attention, (Tensor *[]){rand_tensor(5, 4, true), rand_tensor(4, 4, true), rand_tensor(4, 4, true), rand_tensor(4, 4, true)}, 4); }

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_matmul_forward);
    RUN_TEST(test_transpose_forward);
    RUN_TEST(test_gather_forward);
    RUN_TEST(test_rmsnorm_forward);
    RUN_TEST(test_causal_softmax_forward);
    RUN_TEST(test_softmax_large_logits);
    RUN_TEST(test_relu_scale_forward);
    RUN_TEST(test_slice_concat_forward);
    RUN_TEST(test_cross_entropy_forward);
    RUN_TEST(test_add_grad);
    RUN_TEST(test_add_self_grad);
    RUN_TEST(test_matmul_grad);
    RUN_TEST(test_matmul_frozen_grad);
    RUN_TEST(test_transpose_grad);
    RUN_TEST(test_gather_grad);
    RUN_TEST(test_chain_grad);
    RUN_TEST(test_scale_grad);
    RUN_TEST(test_relu_grad);
    RUN_TEST(test_rmsnorm_grad);
    RUN_TEST(test_causal_mask_grad);
    RUN_TEST(test_softmax_rows_grad);
    RUN_TEST(test_split_merge_grad);
    RUN_TEST(test_cross_entropy_grad);
    RUN_TEST(test_mlp_grad);
    RUN_TEST(test_softmax_of_matmul_grad);
    RUN_TEST(test_attention_grad);
    return UNITY_END();
}
