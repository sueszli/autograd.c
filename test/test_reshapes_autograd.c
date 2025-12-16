#include "autograd.h"
#include "ops/arithmetic.h"
#include "ops/reductions.h"
#include "ops/reshapes.h"
#include "tensor.h"
#include "unity.h"
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

void test_reshape_backward_simple_1d_to_2d(void) {
    uint64_t shape[] = {6};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 1, true);
    int64_t new_shape[] = {2, 3};
    Tensor *y = tensor_reshape(x, new_shape, 2);

    Tensor *scalar = tensor_sum(y, 0, false);
    Tensor *loss = tensor_sum(scalar, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(scalar);
    tensor_release(loss);
}

void test_reshape_backward_2d_to_1d(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    int64_t new_shape[] = {6};
    Tensor *y = tensor_reshape(x, new_shape, 1);
    Tensor *loss = tensor_sum(y, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(loss);
}

void test_reshape_backward_identity(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    int64_t new_shape[] = {2, 3};
    Tensor *y = tensor_reshape(x, new_shape, 2);

    Tensor *scalar = tensor_sum(y, 0, false);
    Tensor *loss = tensor_sum(scalar, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(scalar);
    tensor_release(loss);
}

void test_reshape_backward_3d_to_1d(void) {
    uint64_t shape[] = {2, 2, 2};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    Tensor *x = tensor_create(data, shape, 3, true);
    int64_t new_shape[] = {8};
    Tensor *y = tensor_reshape(x, new_shape, 1);
    Tensor *loss = tensor_sum(y, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(loss);
}

void test_reshape_backward_1d_to_3d(void) {
    uint64_t shape[] = {8};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    Tensor *x = tensor_create(data, shape, 1, true);
    int64_t new_shape[] = {2, 2, 2};
    Tensor *y = tensor_reshape(x, new_shape, 3);

    Tensor *temp = tensor_sum(y, 0, false);
    Tensor *temp2 = tensor_sum(temp, 0, false);
    Tensor *loss = tensor_sum(temp2, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(temp);
    tensor_release(temp2);
    tensor_release(loss);
}

void test_reshape_backward_inferred_dim(void) {
    uint64_t shape[] = {6};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 1, true);
    int64_t new_shape[] = {-1, 2};
    Tensor *y = tensor_reshape(x, new_shape, 2);

    Tensor *scalar = tensor_sum(y, 0, false);
    Tensor *loss = tensor_sum(scalar, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(scalar);
    tensor_release(loss);
}

void test_reshape_backward_chain_with_add(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    int64_t new_shape[] = {6};
    Tensor *y = tensor_reshape(x, new_shape, 1);

    uint64_t shape_const[] = {6};
    float32_t const_data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    Tensor *c = tensor_create(const_data, shape_const, 1, false);
    Tensor *z = tensor_add(y, c);

    Tensor *loss = tensor_sum(z, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(c);
    tensor_release(z);
    tensor_release(loss);
}

void test_reshape_backward_chain_with_mul(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    int64_t new_shape[] = {6};
    Tensor *y = tensor_reshape(x, new_shape, 1);

    uint64_t shape_const[] = {6};
    float32_t const_data[] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    Tensor *c = tensor_create(const_data, shape_const, 1, false);
    Tensor *z = tensor_mul(y, c);

    Tensor *loss = tensor_sum(z, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(c);
    tensor_release(z);
    tensor_release(loss);
}

void test_reshape_backward_double_reshape(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    int64_t new_shape1[] = {6};
    Tensor *y = tensor_reshape(x, new_shape1, 1);
    int64_t new_shape2[] = {2, 3};
    Tensor *z = tensor_reshape(y, new_shape2, 2);

    Tensor *scalar = tensor_sum(z, 0, false);
    Tensor *loss = tensor_sum(scalar, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
    tensor_release(scalar);
    tensor_release(loss);
}

void test_transpose_backward_2d_0_1(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_transpose(x, 0, 1);

    Tensor *scalar = tensor_sum(y, 0, false);
    Tensor *loss = tensor_sum(scalar, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(scalar);
    tensor_release(loss);
}

void test_transpose_backward_double_transpose(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_transpose(x, 0, 1);
    Tensor *z = tensor_transpose(y, 0, 1);

    Tensor *scalar = tensor_sum(z, 0, false);
    Tensor *loss = tensor_sum(scalar, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
    tensor_release(scalar);
    tensor_release(loss);
}

void test_transpose_backward_3d_0_1(void) {
    uint64_t shape[] = {2, 3, 4};
    float32_t *data = malloc(24 * sizeof(float32_t));
    for (uint64_t i = 0; i < 24; i++) {
        data[i] = (float32_t)(i + 1);
    }

    Tensor *x = tensor_create(data, shape, 3, true);
    Tensor *y = tensor_transpose(x, 0, 1);

    Tensor *temp = tensor_sum(y, 0, false);
    Tensor *temp2 = tensor_sum(temp, 0, false);
    Tensor *loss = tensor_sum(temp2, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    free(data);
    tensor_release(x);
    tensor_release(y);
    tensor_release(temp);
    tensor_release(temp2);
    tensor_release(loss);
}

void test_transpose_backward_3d_0_2(void) {
    uint64_t shape[] = {2, 3, 4};
    float32_t *data = malloc(24 * sizeof(float32_t));
    for (uint64_t i = 0; i < 24; i++) {
        data[i] = (float32_t)(i + 1);
    }

    Tensor *x = tensor_create(data, shape, 3, true);
    Tensor *y = tensor_transpose(x, 0, 2);

    Tensor *temp = tensor_sum(y, 0, false);
    Tensor *temp2 = tensor_sum(temp, 0, false);
    Tensor *loss = tensor_sum(temp2, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    free(data);
    tensor_release(x);
    tensor_release(y);
    tensor_release(temp);
    tensor_release(temp2);
    tensor_release(loss);
}

void test_transpose_backward_chain_with_add(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_transpose(x, 0, 1);

    uint64_t shape_const[] = {3, 2};
    float32_t const_data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    Tensor *c = tensor_create(const_data, shape_const, 2, false);
    Tensor *z = tensor_add(y, c);

    Tensor *scalar = tensor_sum(z, 0, false);
    Tensor *loss = tensor_sum(scalar, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(c);
    tensor_release(z);
    tensor_release(scalar);
    tensor_release(loss);
}

void test_transpose_backward_chain_with_mul(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_transpose(x, 0, 1);

    uint64_t shape_const[] = {3, 2};
    float32_t const_data[] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    Tensor *c = tensor_create(const_data, shape_const, 2, false);
    Tensor *z = tensor_mul(y, c);

    Tensor *scalar = tensor_sum(z, 0, false);
    Tensor *loss = tensor_sum(scalar, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 2.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(c);
    tensor_release(z);
    tensor_release(scalar);
    tensor_release(loss);
}

void test_reshape_transpose_chain(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    int64_t new_shape[] = {6};
    Tensor *y = tensor_reshape(x, new_shape, 1);
    int64_t new_shape2[] = {3, 2};
    Tensor *z = tensor_reshape(y, new_shape2, 2);
    Tensor *w = tensor_transpose(z, 0, 1);

    Tensor *scalar = tensor_sum(w, 0, false);
    Tensor *loss = tensor_sum(scalar, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
    tensor_release(w);
    tensor_release(scalar);
    tensor_release(loss);
}

void test_transpose_reshape_chain(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_transpose(x, 0, 1);
    int64_t new_shape[] = {6};
    Tensor *z = tensor_reshape(y, new_shape, 1);

    Tensor *loss = tensor_sum(z, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
    tensor_release(loss);
}

void test_reshape_backward_with_mean(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    int64_t new_shape[] = {6};
    Tensor *y = tensor_reshape(x, new_shape, 1);
    Tensor *z = tensor_mean(y, 0, false);

    backward(z);

    TEST_ASSERT_NOT_NULL(x->grad);
    float32_t expected_grad = 1.0f / 6.0f;
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_grad, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
}

void test_transpose_backward_with_mean(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_transpose(x, 0, 1);
    Tensor *z = tensor_mean(y, 0, false);
    Tensor *loss = tensor_sum(z, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    float32_t expected_grad = 1.0f / 3.0f;
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_grad, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
    tensor_release(loss);
}

void test_reshape_backward_scalar(void) {
    uint64_t shape[] = {1};
    float32_t data[] = {5.0f};

    Tensor *x = tensor_create(data, shape, 1, true);
    int64_t new_shape[] = {1, 1};
    Tensor *y = tensor_reshape(x, new_shape, 2);

    Tensor *temp = tensor_sum(y, 0, false);
    Tensor *loss = tensor_sum(temp, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[0]);

    tensor_release(x);
    tensor_release(y);
    tensor_release(temp);
    tensor_release(loss);
}

void test_transpose_backward_identity_dims(void) {
    uint64_t shape[] = {2, 3};
    float32_t data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    Tensor *x = tensor_create(data, shape, 2, true);
    Tensor *y = tensor_transpose(x, 0, 0);

    Tensor *scalar = tensor_sum(y, 0, false);
    Tensor *loss = tensor_sum(scalar, 0, false);

    backward(loss);

    TEST_ASSERT_NOT_NULL(x->grad);
    for (uint64_t i = 0; i < x->size; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[i]);
    }

    tensor_release(x);
    tensor_release(y);
    tensor_release(scalar);
    tensor_release(loss);
}

void test_reshape_backward_large_dims(void) {
    uint64_t shape[] = {100, 100};
    uint64_t size = 10000;
    float32_t *data = malloc(size * sizeof(float32_t));
    for(int i=0; i<size; i++) data[i] = 1.0f;
    
    Tensor *x = tensor_create(data, shape, 2, true);
    int64_t new_shape[] = {10000};
    Tensor *y = tensor_reshape(x, new_shape, 1);
    
    Tensor *loss = tensor_sum(y, 0, false);
    backward(loss);
    
    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_EQUAL_UINT64(2, x->grad->ndim);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[size-1]);
    
    free(data);
    tensor_release(x);
    tensor_release(y);
    tensor_release(loss);
}

void test_transpose_backward_high_dims(void) {
    uint64_t shape[] = {2, 2, 2, 2};
    float32_t data[16] = {0};
    Tensor *x = tensor_create(data, shape, 4, true);
    
    Tensor *y = tensor_transpose(x, 1, 2);
    Tensor *z = tensor_transpose(y, 0, 3);
    
    Tensor *scalar = tensor_sum(z, 0, false);
    Tensor *scalar2 = tensor_sum(scalar, 0, false);
    Tensor *scalar3 = tensor_sum(scalar2, 0, false);
    Tensor *loss = tensor_sum(scalar3, 0, false);
    
    backward(loss);
    
    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[0]);
    
    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
    tensor_release(scalar);
    tensor_release(scalar2);
    tensor_release(scalar3);
    tensor_release(loss);
}

void test_reshape_cycle(void) {
    uint64_t shape[] = {2, 2};
    float32_t data[] = {1, 2, 3, 4};
    Tensor *x = tensor_create(data, shape, 2, true);
    
    int64_t shape1[] = {4};
    Tensor *y = tensor_reshape(x, shape1, 1);
    
    int64_t shape2[] = {2, 2};
    Tensor *z = tensor_reshape(y, shape2, 2);
    
    Tensor *loss = tensor_sum(z, 0, false); 
    Tensor *final_loss = tensor_sum(loss, 0, false);
    
    backward(final_loss);
    
    TEST_ASSERT_NOT_NULL(x->grad);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, x->grad->data[0]);
    
    tensor_release(x);
    tensor_release(y);
    tensor_release(z);
    tensor_release(loss);
    tensor_release(final_loss);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_reshape_backward_simple_1d_to_2d);
    RUN_TEST(test_reshape_backward_2d_to_1d);
    RUN_TEST(test_reshape_backward_identity);
    RUN_TEST(test_reshape_backward_3d_to_1d);
    RUN_TEST(test_reshape_backward_1d_to_3d);
    RUN_TEST(test_reshape_backward_inferred_dim);
    RUN_TEST(test_reshape_backward_chain_with_add);
    RUN_TEST(test_reshape_backward_chain_with_mul);
    RUN_TEST(test_reshape_backward_double_reshape);
    RUN_TEST(test_transpose_backward_2d_0_1);
    RUN_TEST(test_transpose_backward_double_transpose);
    RUN_TEST(test_transpose_backward_3d_0_1);
    RUN_TEST(test_transpose_backward_3d_0_2);
    RUN_TEST(test_transpose_backward_chain_with_add);
    RUN_TEST(test_transpose_backward_chain_with_mul);
    RUN_TEST(test_reshape_transpose_chain);
    RUN_TEST(test_transpose_reshape_chain);
    RUN_TEST(test_reshape_backward_with_mean);
    RUN_TEST(test_transpose_backward_with_mean);
    RUN_TEST(test_reshape_backward_scalar);
    RUN_TEST(test_transpose_backward_identity_dims);
    RUN_TEST(test_reshape_backward_large_dims);
    RUN_TEST(test_transpose_backward_high_dims);
    RUN_TEST(test_reshape_cycle);
    return UNITY_END();
}
