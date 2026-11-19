#include "autograd.h"
#include "ops.h"
#include "std/assert.h"
#include "std/random.h"
#include "unity.h"
#include <math.h>

void setUp(void) {}
void tearDown(void) {
    graph_reset();
    perm_reset();
}

static u64 rng = 1;

static Tensor *rand_tensor(const i64 *shape, i64 ndim, bool requires_grad) {
    Tensor *t = tensor_new(shape, ndim, NULL, requires_grad);
    random_uniform(&rng, t->data, t->size, -1.0, 1.0);
    return t;
}

/*
 * reduces a matrix f(in) to a scalar with fixed random vectors, loss = r1 @ f(in) @ r2.
 * a scalar f(in) is the loss itself.
 * then checks the analytic gradient of every input element with assert_grad.
 */
typedef Tensor *(*Fn)(Tensor **in);

typedef struct {
    Fn f;
    Tensor **in, *r1, *r2;
} Probe;

static Tensor *reduce(Tensor *out, Tensor *r1, Tensor *r2) { return out->ndim == 0 ? out : matmul(matmul(r1, out), r2); }

static f64 loss_of(void *ctx) {
    const Probe *p = ctx;
    f64 loss = reduce(p->f(p->in), p->r1, p->r2)->data[0];
    graph_reset();
    return loss;
}

static void gradcheck(Fn f, Tensor **in, i64 n_in) {
    Tensor *out = f(in);
    Tensor *r1 = out->ndim == 0 ? NULL : rand_tensor(shape(1, out->shape[0]), false);
    Tensor *r2 = out->ndim == 0 ? NULL : rand_tensor(shape(out->shape[1], 1), false);
    // grads start at 0.5, so a backward that overwrites instead of accumulating fails
    for (i64 t = 0; t < n_in; t++) {
        for (i64 i = 0; in[t]->requires_grad && i < in[t]->size; i++) {
            in[t]->grad[i] = 0.5;
        }
    }
    backward(reduce(out, r1, r2));
    graph_reset();
    Probe probe = {f, in, r1, r2};
    for (i64 t = 0; t < n_in; t++) {
        for (i64 i = 0; in[t]->requires_grad && i < in[t]->size; i++) {
            in[t]->grad[i] -= 0.5;
        }
        if (in[t]->requires_grad) {
            assert_grad(loss_of, &probe, in[t]->data, in[t]->grad, in[t]->size, 1e-6, 1e-6, "grad");
        }
    }
}

//
// forward values
//

void test_matmul_forward(void) {
    Tensor *a = tensor_new(shape(2, 2), (f64[]){1, 2, 3, 4}, false);
    Tensor *b = tensor_new(shape(2, 1), (f64[]){5, 6}, false);
    Tensor *out = matmul(a, b);
    TEST_ASSERT_EQUAL_DOUBLE(17, out->data[0]);
    TEST_ASSERT_EQUAL_DOUBLE(39, out->data[1]);
}

void test_batched_matmul_forward(void) {
    Tensor *a = tensor_new(shape(2, 1, 2), (f64[]){1, 2, 3, 4}, false);
    Tensor *b = tensor_new(shape(2, 1), (f64[]){5, 6}, false);
    Tensor *out = matmul(a, b);
    TEST_ASSERT_EQUAL_INT64(3, out->ndim);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((f64[]){17, 39}), out->data, 2);
}

void test_transpose_forward(void) {
    Tensor *a = tensor_new(shape(2, 3), (f64[]){1, 2, 3, 4, 5, 6}, false);
    Tensor *out = transpose(a);
    TEST_ASSERT_EQUAL_INT64(3, out->shape[0]);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((f64[]){1, 4, 2, 5, 3, 6}), out->data, 6);
}

void test_embedding_forward(void) {
    Tensor *table = tensor_new(shape(3, 2), (f64[]){1, 2, 3, 4, 5, 6}, false);
    Tensor *out = embedding((i64[]){2, 0, 2}, 3, table);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((f64[]){5, 6, 1, 2, 5, 6}), out->data, 6);
}

void test_rms_norm_forward(void) {
    Tensor *a = tensor_new(shape(1, 2), (f64[]){3, 4}, false);
    Tensor *out = rms_norm(a, 0.0);
    // rms of (3, 4) is sqrt(12.5)
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, 3 / sqrt(12.5), out->data[0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, 4 / sqrt(12.5), out->data[1]);
}

void test_masked_softmax_forward(void) {
    Tensor *a = tensor_new(shape(2, 2), (f64[]){1, 5, 2, 2}, false);
    Tensor *out = softmax(masked_fill_upper(a, -1e9));
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((f64[]){1, 0, 0.5, 0.5}), out->data, 4);
}

// large logits overflow exp unless each row subtracts its max first
void test_softmax_large_logits(void) {
    Tensor *a = tensor_new(shape(2, 2), (f64[]){1000, 0, 0, 1000}, false);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((f64[]){1, 0, 0, 1}), softmax(a)->data, 4);
}

void test_relu_mul_forward(void) {
    Tensor *a = tensor_new(shape(1, 3), (f64[]){-1, 0, 2}, false);
    Tensor *out = mul(relu(a), 0.5);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((f64[]){0, 0, 1}), out->data, 3);
}

void test_narrow_cat_forward(void) {
    Tensor *a = tensor_new(shape(2, 3), (f64[]){1, 2, 3, 4, 5, 6}, false);
    Tensor *out = cat((Tensor *[]){narrow(a, 2, 1), narrow(a, 0, 2)}, 2);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((f64[]){3, 1, 2, 6, 4, 5}), out->data, 6);
}

void test_cross_entropy_forward(void) {
    // uniform logits give log(cols) per row, the masked row is ignored
    Tensor *logits = tensor_new(shape(2, 4), (f64[]){0, 0, 0, 0, 9, 1, 2, 3}, false);
    Tensor *out = cross_entropy(logits, (i64[]){1, 2}, (f64[]){1, 0});
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, log(4.0), out->data[0]);
    Tensor *none = cross_entropy(logits, (i64[]){1, 2}, (f64[]){0, 0});
    TEST_ASSERT_EQUAL_DOUBLE(0.0, none->data[0]);
}

//
// gradients
//

static Tensor *f_add(Tensor **in) { return add(in[0], in[1]); }
static Tensor *f_add_self(Tensor **in) { return add(in[0], in[0]); }
static Tensor *f_matmul(Tensor **in) { return matmul(in[0], in[1]); }
static Tensor *f_transpose(Tensor **in) { return transpose(in[0]); }
static Tensor *f_embedding(Tensor **in) { return embedding((i64[]){1, 0, 1, 3}, 4, in[0]); }
static Tensor *f_chain(Tensor **in) { return matmul(transpose(matmul(in[0], in[1])), in[2]); }
static Tensor *f_mul(Tensor **in) { return mul(in[0], -0.3); }
static Tensor *f_relu(Tensor **in) { return relu(in[0]); }
static Tensor *f_rms_norm(Tensor **in) { return rms_norm(in[0], 1e-5); }
// a small mask value, -1e9 would make the finite difference lose all precision
static Tensor *f_masked_fill_upper(Tensor **in) { return masked_fill_upper(in[0], -5.0); }
static Tensor *f_softmax(Tensor **in) { return softmax(in[0]); }
static Tensor *f_narrow_cat(Tensor **in) { return cat((Tensor *[]){narrow(in[0], 3, 1), narrow(in[0], 0, 3), in[1]}, 3); }
static Tensor *f_cross_entropy(Tensor **in) { return cross_entropy(in[0], (i64[]){2, 0, 4}, (f64[]){1, 0, 1}); }

// non-leaf inputs to matmul and softmax once lost their gradients
static Tensor *f_mlp(Tensor **in) { return matmul(relu(matmul(in[0], in[1])), in[2]); }
static Tensor *f_softmax_of_matmul(Tensor **in) { return softmax(matmul(in[0], in[1])); }

// 3-d inputs, matmul, transpose and the mask batch over the leading dim, the rest act on rows
static Tensor *f_batched(Tensor **in) {
    Tensor *scores = masked_fill_upper(transpose(matmul(in[0], in[1])), -5.0);
    Tensor *h = cat((Tensor *[]){narrow(rms_norm(scores, 1e-5), 0, 2), narrow(softmax(scores), 2, 1)}, 2);
    return cross_entropy(add(h, mul(h, 0.5)), (i64[]){0, 1, 2, 2, 1, 0}, (f64[]){1, 1, 0, 1, 1, 1});
}

// one causal attention head: softmax(mask(q @ k^T / sqrt(d))) @ v
static Tensor *f_attention(Tensor **in) {
    Tensor *xn = rms_norm(in[0], 1e-5);
    Tensor *q = matmul(xn, in[1]), *k = matmul(xn, in[2]), *v = matmul(xn, in[3]);
    return matmul(softmax(masked_fill_upper(mul(matmul(q, transpose(k)), 0.5), -1e9)), v);
}

void test_add_grad(void) { gradcheck(f_add, (Tensor *[]){rand_tensor(shape(3, 4), true), rand_tensor(shape(3, 4), true)}, 2); }
void test_add_self_grad(void) { gradcheck(f_add_self, (Tensor *[]){rand_tensor(shape(3, 4), true)}, 1); }
void test_matmul_grad(void) { gradcheck(f_matmul, (Tensor *[]){rand_tensor(shape(3, 4), true), rand_tensor(shape(4, 5), true)}, 2); }
void test_matmul_frozen_grad(void) { gradcheck(f_matmul, (Tensor *[]){rand_tensor(shape(3, 4), false), rand_tensor(shape(4, 5), true)}, 2); }
void test_transpose_grad(void) { gradcheck(f_transpose, (Tensor *[]){rand_tensor(shape(3, 4), true)}, 1); }
void test_embedding_grad(void) { gradcheck(f_embedding, (Tensor *[]){rand_tensor(shape(5, 3), true)}, 1); }
void test_chain_grad(void) { gradcheck(f_chain, (Tensor *[]){rand_tensor(shape(3, 4), true), rand_tensor(shape(4, 2), true), rand_tensor(shape(3, 5), true)}, 3); }
void test_mul_grad(void) { gradcheck(f_mul, (Tensor *[]){rand_tensor(shape(3, 4), true)}, 1); }
void test_relu_grad(void) { gradcheck(f_relu, (Tensor *[]){rand_tensor(shape(3, 4), true)}, 1); }
void test_rms_norm_grad(void) { gradcheck(f_rms_norm, (Tensor *[]){rand_tensor(shape(3, 4), true)}, 1); }
void test_masked_fill_upper_grad(void) { gradcheck(f_masked_fill_upper, (Tensor *[]){rand_tensor(shape(4, 4), true)}, 1); }
void test_softmax_grad(void) { gradcheck(f_softmax, (Tensor *[]){rand_tensor(shape(3, 4), true)}, 1); }
void test_narrow_cat_grad(void) { gradcheck(f_narrow_cat, (Tensor *[]){rand_tensor(shape(3, 4), true), rand_tensor(shape(3, 2), true)}, 2); }
void test_cross_entropy_grad(void) { gradcheck(f_cross_entropy, (Tensor *[]){rand_tensor(shape(3, 5), true)}, 1); }
void test_mlp_grad(void) { gradcheck(f_mlp, (Tensor *[]){rand_tensor(shape(3, 4), false), rand_tensor(shape(4, 8), true), rand_tensor(shape(8, 2), true)}, 3); }
void test_softmax_of_matmul_grad(void) { gradcheck(f_softmax_of_matmul, (Tensor *[]){rand_tensor(shape(3, 4), false), rand_tensor(shape(4, 5), true)}, 2); }
void test_batched_grad(void) { gradcheck(f_batched, (Tensor *[]){rand_tensor(shape(2, 3, 3), true), rand_tensor(shape(3, 3), true)}, 2); }
void test_attention_grad(void) { gradcheck(f_attention, (Tensor *[]){rand_tensor(shape(5, 4), true), rand_tensor(shape(4, 4), true), rand_tensor(shape(4, 4), true), rand_tensor(shape(4, 4), true)}, 4); }

i32 main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_matmul_forward);
    RUN_TEST(test_batched_matmul_forward);
    RUN_TEST(test_transpose_forward);
    RUN_TEST(test_embedding_forward);
    RUN_TEST(test_rms_norm_forward);
    RUN_TEST(test_masked_softmax_forward);
    RUN_TEST(test_softmax_large_logits);
    RUN_TEST(test_relu_mul_forward);
    RUN_TEST(test_narrow_cat_forward);
    RUN_TEST(test_cross_entropy_forward);
    RUN_TEST(test_add_grad);
    RUN_TEST(test_add_self_grad);
    RUN_TEST(test_matmul_grad);
    RUN_TEST(test_matmul_frozen_grad);
    RUN_TEST(test_transpose_grad);
    RUN_TEST(test_embedding_grad);
    RUN_TEST(test_chain_grad);
    RUN_TEST(test_mul_grad);
    RUN_TEST(test_relu_grad);
    RUN_TEST(test_rms_norm_grad);
    RUN_TEST(test_masked_fill_upper_grad);
    RUN_TEST(test_softmax_grad);
    RUN_TEST(test_narrow_cat_grad);
    RUN_TEST(test_cross_entropy_grad);
    RUN_TEST(test_mlp_grad);
    RUN_TEST(test_softmax_of_matmul_grad);
    RUN_TEST(test_batched_grad);
    RUN_TEST(test_attention_grad);
    return UNITY_END();
}
