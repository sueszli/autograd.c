#include "ops/reshapes.h"
#include "tensor.h"
#include "unity.h"
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

static Tensor *create_ranged_tensor(uint64_t *shape, uint64_t ndim) {
    uint64_t size = 1;
    for (uint64_t i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    float32_t *data = malloc(size * sizeof(float32_t));
    for (uint64_t i = 0; i < size; i++) {
        data[i] = (float32_t)i;
    }
    Tensor *t = tensor_create(data, shape, ndim, false);
    free(data);
    return t;
}

void test_reshape_identity(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = create_ranged_tensor(shape, 2);
    int64_t new_shape[] = {2, 3};
    Tensor *r = tensor_reshape(t, new_shape, 2);

    TEST_ASSERT_EQUAL_UINT64(2, r->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, r->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, r->shape[1]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, r->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, r->data[5]);

    tensor_free(t);
    tensor_free(r);
}

void test_reshape_flatten(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = create_ranged_tensor(shape, 2);
    int64_t new_shape[] = {6};
    Tensor *r = tensor_reshape(t, new_shape, 1);

    TEST_ASSERT_EQUAL_UINT64(1, r->ndim);
    TEST_ASSERT_EQUAL_UINT64(6, r->shape[0]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, r->data[5]);

    tensor_free(t);
    tensor_free(r);
}

void test_reshape_extend_rank(void) {
    uint64_t shape[] = {6};
    Tensor *t = create_ranged_tensor(shape, 1);
    int64_t new_shape[] = {1, 2, 3};
    Tensor *r = tensor_reshape(t, new_shape, 3);

    TEST_ASSERT_EQUAL_UINT64(3, r->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, r->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, r->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(3, r->shape[2]);

    tensor_free(t);
    tensor_free(r);
}

void test_reshape_infer_dim_0(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = create_ranged_tensor(shape, 2);
    int64_t new_shape[] = {-1, 2};
    Tensor *r = tensor_reshape(t, new_shape, 2);

    TEST_ASSERT_EQUAL_UINT64(2, r->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, r->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, r->shape[1]);

    tensor_free(t);
    tensor_free(r);
}

void test_reshape_infer_dim_1(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = create_ranged_tensor(shape, 2);
    int64_t new_shape[] = {2, -1};
    Tensor *r = tensor_reshape(t, new_shape, 2);

    TEST_ASSERT_EQUAL_UINT64(2, r->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, r->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, r->shape[1]);

    tensor_free(t);
    tensor_free(r);
}

void test_reshape_scalar_expansion(void) {
    uint64_t shape[] = {1};
    Tensor *t = create_ranged_tensor(shape, 0);
    TEST_ASSERT_EQUAL_UINT64(0, t->ndim);

    int64_t new_shape[] = {1, 1};
    Tensor *r = tensor_reshape(t, new_shape, 2);

    TEST_ASSERT_EQUAL_UINT64(2, r->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, r->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(1, r->shape[1]);

    tensor_free(t);
    tensor_free(r);
}

void test_transpose_identity_swap(void) {
    uint64_t shape[] = {2, 2};
    Tensor *t = create_ranged_tensor(shape, 2);
    Tensor *r = tensor_transpose(t, 0, 0);

    TEST_ASSERT_EQUAL_FLOAT(t->data[1], r->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(t->data[2], r->data[2]);

    tensor_free(t);
    tensor_free(r);
}

void test_transpose_vector(void) {
    uint64_t shape[] = {5};
    Tensor *t = create_ranged_tensor(shape, 1);
    Tensor *r = tensor_transpose(t, 0, 0);

    TEST_ASSERT_EQUAL_UINT64(1, r->ndim);
    TEST_ASSERT_EQUAL_UINT64(5, r->shape[0]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, r->data[4]);

    tensor_free(t);
    tensor_free(r);
}

void test_transpose_general(void) {
    uint64_t shape[] = {2, 3};
    /*
     0 1 2
     3 4 5
     */
    Tensor *t = create_ranged_tensor(shape, 2);
    Tensor *r = tensor_transpose(t, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(2, r->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, r->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, r->shape[1]);

    /*
     0 3
     1 4
     2 5
     */
    TEST_ASSERT_EQUAL_FLOAT(0.0f, r->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, r->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, r->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, r->data[3]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, r->data[4]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, r->data[5]);

    tensor_free(t);
    tensor_free(r);
}

void test_transpose_3d_0_1(void) {
    uint64_t shape[] = {2, 2, 2};
    Tensor *t = create_ranged_tensor(shape, 3);
    /*
     t[0]:
       0 1
       2 3
     t[1]:
       4 5
       6 7
     */
    Tensor *r = tensor_transpose(t, 0, 1);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, r->data[4]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, r->data[3]);

    tensor_free(t);
    tensor_free(r);
}

void test_transpose_3d_0_2(void) {
    uint64_t shape[] = {2, 3, 4};
    Tensor *t = create_ranged_tensor(shape, 3);
    Tensor *r = tensor_transpose(t, 0, 2);

    TEST_ASSERT_EQUAL_UINT64(3, r->ndim);
    TEST_ASSERT_EQUAL_UINT64(4, r->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, r->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(2, r->shape[2]);

    tensor_free(t);
    tensor_free(r);
}

void test_transpose_3d_1_2(void) {
    uint64_t shape[] = {2, 3, 4};
    Tensor *t = create_ranged_tensor(shape, 3);
    Tensor *r = tensor_transpose(t, 1, 2);

    TEST_ASSERT_EQUAL_UINT64(3, r->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, r->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(4, r->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(3, r->shape[2]);

    tensor_free(t);
    tensor_free(r);
}

void test_transpose_double(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = create_ranged_tensor(shape, 2);
    Tensor *t1 = tensor_transpose(t, 0, 1);
    Tensor *t2 = tensor_transpose(t1, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(t->ndim, t2->ndim);
    TEST_ASSERT_EQUAL_UINT64(t->shape[0], t2->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(t->shape[1], t2->shape[1]);
    for (int i = 0; i < 6; ++i) {
        TEST_ASSERT_EQUAL_FLOAT(t->data[i], t2->data[i]);
    }
    tensor_free(t);
    tensor_free(t1);
    tensor_free(t2);
}

void test_transpose_4d_0_3(void) {
    uint64_t shape[] = {2, 2, 2, 2};
    Tensor *t = create_ranged_tensor(shape, 4);
    Tensor *r = tensor_transpose(t, 0, 3);

    TEST_ASSERT_EQUAL_UINT64(4, r->ndim);
    TEST_ASSERT_EQUAL_FLOAT(t->data[8], r->data[1]);

    tensor_free(t);
    tensor_free(r);
}

void test_chain_reshape_transpose(void) {
    uint64_t shape[] = {6};
    Tensor *t = create_ranged_tensor(shape, 1);
    int64_t new_shape[] = {2, 3};
    Tensor *r1 = tensor_reshape(t, new_shape, 2);
    Tensor *r2 = tensor_transpose(r1, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(2, r2->ndim);
    TEST_ASSERT_EQUAL_UINT64(3, r2->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, r2->shape[1]);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, r2->data[1]);

    tensor_free(t);
    tensor_free(r1);
    tensor_free(r2);
}

void test_chain_transpose_reshape(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = create_ranged_tensor(shape, 2);
    Tensor *r1 = tensor_transpose(t, 0, 1);
    int64_t new_shape[] = {6};
    Tensor *r2 = tensor_reshape(r1, new_shape, 1);

    TEST_ASSERT_EQUAL_FLOAT(3.0f, r2->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, r2->data[2]);

    tensor_free(t);
    tensor_free(r1);
    tensor_free(r2);
}

void test_reshape_to_scalar(void) {
    uint64_t shape[] = {1, 1};
    Tensor *t = create_ranged_tensor(shape, 2);
    int64_t new_shape[] = {1};
    Tensor *r = tensor_reshape(t, new_shape, 0);

    TEST_ASSERT_EQUAL_UINT64(0, r->ndim);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, r->data[0]);

    tensor_free(t);
    tensor_free(r);
}

void test_reshape_infer_last(void) {
    uint64_t shape[] = {2, 2};
    Tensor *t = create_ranged_tensor(shape, 2);
    int64_t new_shape[] = {4, -1};
    Tensor *r = tensor_reshape(t, new_shape, 2);

    TEST_ASSERT_EQUAL_UINT64(4, r->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(1, r->shape[1]);

    tensor_free(t);
    tensor_free(r);
}

void test_reshape_infer_first(void) {
    uint64_t shape[] = {2, 2};
    Tensor *t = create_ranged_tensor(shape, 2);
    int64_t new_shape[] = {-1, 4};
    Tensor *r = tensor_reshape(t, new_shape, 2);

    TEST_ASSERT_EQUAL_UINT64(1, r->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(4, r->shape[1]);

    tensor_free(t);
    tensor_free(r);
}

void test_reshape_infer_middle(void) {
    uint64_t shape[] = {2, 3, 4};
    Tensor *t = create_ranged_tensor(shape, 3);
    int64_t new_shape[] = {2, -1, 3};
    Tensor *r = tensor_reshape(t, new_shape, 3);

    TEST_ASSERT_EQUAL_UINT64(2, r->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(4, r->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(3, r->shape[2]);

    tensor_free(t);
    tensor_free(r);
}

void test_reshape_large(void) {
    uint64_t shape[] = {100, 100};
    Tensor *t = tensor_zeros(shape, 2, false);
    int64_t new_shape[] = {10000};
    Tensor *r = tensor_reshape(t, new_shape, 1);

    TEST_ASSERT_EQUAL_UINT64(1, r->ndim);
    TEST_ASSERT_EQUAL_UINT64(10000, r->shape[0]);

    tensor_free(t);
    tensor_free(r);
}

void test_transpose_row_to_col(void) {
    uint64_t shape[] = {1, 5};
    Tensor *t = create_ranged_tensor(shape, 2);
    Tensor *r = tensor_transpose(t, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(5, r->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(1, r->shape[1]);

    tensor_free(t);
    tensor_free(r);
}

void test_transpose_col_to_row(void) {
    uint64_t shape[] = {5, 1};
    Tensor *t = create_ranged_tensor(shape, 2);
    Tensor *r = tensor_transpose(t, 0, 1);

    TEST_ASSERT_EQUAL_UINT64(1, r->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(5, r->shape[1]);

    tensor_free(t);
    tensor_free(r);
}

void test_transpose_square(void) {
    uint64_t shape[] = {3, 3};
    Tensor *t = create_ranged_tensor(shape, 2);
    Tensor *r = tensor_transpose(t, 0, 1);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, r->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, r->data[3]);
    TEST_ASSERT_EQUAL_FLOAT(7.0f, r->data[5]);

    tensor_free(t);
    tensor_free(r);
}

void test_reshape_unsqueeze(void) {
    uint64_t shape[] = {2, 2};
    Tensor *t = create_ranged_tensor(shape, 2);
    int64_t new_shape[] = {1, 2, 1, 2, 1};
    Tensor *r = tensor_reshape(t, new_shape, 5);

    TEST_ASSERT_EQUAL_UINT64(5, r->ndim);
    TEST_ASSERT_EQUAL_UINT64(1, r->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, r->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(1, r->shape[2]);
    TEST_ASSERT_EQUAL_UINT64(2, r->shape[3]);
    TEST_ASSERT_EQUAL_UINT64(1, r->shape[4]);

    tensor_free(t);
    tensor_free(r);
}

void test_transpose_3d_swap_ends(void) {
    uint64_t shape[] = {2, 3, 4};
    Tensor *t = create_ranged_tensor(shape, 3);
    Tensor *r = tensor_transpose(t, 0, 2);

    TEST_ASSERT_EQUAL_FLOAT(23.0f, r->data[23]);

    tensor_free(t);
    tensor_free(r);
}

void test_reshape_same_shape(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = create_ranged_tensor(shape, 2);
    int64_t new_shape[] = {2, 3};
    Tensor *r = tensor_reshape(t, new_shape, 2);

    TEST_ASSERT_NOT_EQUAL(t, r);
    TEST_ASSERT_EQUAL_FLOAT(t->data[0], r->data[0]);

    tensor_free(t);
    tensor_free(r);
}

void test_transpose_same_dims(void) {
    uint64_t shape[] = {2, 3};
    Tensor *t = create_ranged_tensor(shape, 2);
    Tensor *r = tensor_transpose(t, 0, 0);

    TEST_ASSERT_NOT_EQUAL(t, r);
    TEST_ASSERT_EQUAL_UINT64(2, r->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, r->shape[1]);

    tensor_free(t);
    tensor_free(r);
}

void test_reshape_scalar_identity(void) {
    uint64_t shape[] = {1};
    Tensor *t = create_ranged_tensor(shape, 0);
    int64_t dummy_shape[] = {0};
    Tensor *r = tensor_reshape(t, dummy_shape, 0);

    TEST_ASSERT_EQUAL_UINT64(0, r->ndim);
    TEST_ASSERT_EQUAL_FLOAT(t->data[0], r->data[0]);

    tensor_free(t);
    tensor_free(r);
}

void test_reshape_vector_to_scalar(void) {
    uint64_t shape[] = {1};
    Tensor *t = create_ranged_tensor(shape, 1);
    int64_t dummy_shape[] = {0};
    Tensor *r = tensor_reshape(t, dummy_shape, 0);

    TEST_ASSERT_EQUAL_UINT64(0, r->ndim);
    TEST_ASSERT_EQUAL_FLOAT(t->data[0], r->data[0]);

    tensor_free(t);
    tensor_free(r);
}

void test_ops_stress(void) {
    uint64_t shape[] = {2, 2};
    Tensor *t = create_ranged_tensor(shape, 2);

    int64_t s1[] = {4};
    Tensor *t1 = tensor_reshape(t, s1, 1);

    int64_t s2[] = {1, 4};
    Tensor *t2 = tensor_reshape(t1, s2, 2);

    Tensor *t3 = tensor_transpose(t2, 0, 1);

    int64_t s3[] = {2, 2};
    Tensor *t4 = tensor_reshape(t3, s3, 2);

    TEST_ASSERT_EQUAL_FLOAT(0.0f, t4->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, t4->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, t4->data[2]);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, t4->data[3]);

    tensor_free(t);
    tensor_free(t1);
    tensor_free(t2);
    tensor_free(t3);
    tensor_free(t4);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_reshape_identity);
    RUN_TEST(test_reshape_flatten);
    RUN_TEST(test_reshape_extend_rank);
    RUN_TEST(test_reshape_infer_dim_0);
    RUN_TEST(test_reshape_infer_dim_1);
    RUN_TEST(test_reshape_scalar_expansion);
    RUN_TEST(test_transpose_identity_swap);
    RUN_TEST(test_transpose_vector);
    RUN_TEST(test_transpose_general);
    RUN_TEST(test_transpose_3d_0_1);
    RUN_TEST(test_transpose_3d_0_2);
    RUN_TEST(test_transpose_3d_1_2);
    RUN_TEST(test_transpose_double);
    RUN_TEST(test_transpose_4d_0_3);
    RUN_TEST(test_chain_reshape_transpose);
    RUN_TEST(test_chain_transpose_reshape);
    RUN_TEST(test_reshape_to_scalar);
    RUN_TEST(test_reshape_infer_last);
    RUN_TEST(test_reshape_infer_first);
    RUN_TEST(test_reshape_infer_middle);
    RUN_TEST(test_reshape_large);
    RUN_TEST(test_transpose_row_to_col);
    RUN_TEST(test_transpose_col_to_row);
    RUN_TEST(test_transpose_square);
    RUN_TEST(test_reshape_unsqueeze);
    RUN_TEST(test_transpose_3d_swap_ends);
    RUN_TEST(test_reshape_same_shape);
    RUN_TEST(test_transpose_same_dims);
    RUN_TEST(test_reshape_scalar_identity);
    RUN_TEST(test_reshape_vector_to_scalar);
    RUN_TEST(test_ops_stress);
    return UNITY_END();
}
