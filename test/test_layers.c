#include "../src/layers.h"
#include "../src/tensor.h"
#include "unity.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>

void setUp(void) { srand(42); }
void tearDown(void) {}

static Tensor *create_tensor_from_data(float32_t *data, uint64_t *shape, uint64_t ndim) { return tensor_create(data, shape, ndim, false); }

void test_linear_creation(void) {
    Layer *l = layer_linear_create(10, 5, true);
    TEST_ASSERT_NOT_NULL(l);

    Tensor **params;
    size_t count;
    layer_parameters(l, &params, &count);

    TEST_ASSERT_EQUAL_UINT64(2, count);
    TEST_ASSERT_NOT_NULL(params[0]);
    TEST_ASSERT_NOT_NULL(params[1]);

    TEST_ASSERT_EQUAL_UINT64(2, params[0]->ndim);
    TEST_ASSERT_EQUAL_UINT64(10, params[0]->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(5, params[0]->shape[1]);

    TEST_ASSERT_EQUAL_UINT64(1, params[1]->ndim);
    TEST_ASSERT_EQUAL_UINT64(5, params[1]->shape[0]);

    if (params)
        free(params);
    layer_free(l);
}

void test_linear_creation_no_bias(void) {
    Layer *l = layer_linear_create(10, 5, false);
    TEST_ASSERT_NOT_NULL(l);

    Tensor **params;
    size_t count;
    layer_parameters(l, &params, &count);

    TEST_ASSERT_EQUAL_UINT64(1, count);
    TEST_ASSERT_NOT_NULL(params[0]);

    if (params)
        free(params);
    layer_free(l);
}

void test_linear_forward_shape(void) {
    Layer *l = layer_linear_create(3, 4, true);

    uint64_t in_shape[] = {2, 3};
    Tensor *input = tensor_zeros(in_shape, 2, false);

    Tensor *out = layer_forward(l, input, true);

    TEST_ASSERT_NOT_NULL(out);
    TEST_ASSERT_EQUAL_UINT64(2, out->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, out->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(4, out->shape[1]);

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_linear_forward_values(void) {
    Layer *l = layer_linear_create(2, 2, true);

    Tensor **params;
    size_t count;
    layer_parameters(l, &params, &count);

    Tensor *w = params[0];
    Tensor *b = params[1];

    w->data[0] = 0.5f;
    w->data[1] = -0.5f;
    w->data[2] = 1.0f;
    w->data[3] = 2.0f;

    b->data[0] = 0.1f;
    b->data[1] = 0.2f;

    if (params)
        free(params);

    float32_t in_data[] = {1.0f, 2.0f};
    uint64_t in_shape[] = {1, 2};
    Tensor *input = create_tensor_from_data(in_data, in_shape, 2);

    Tensor *result = layer_forward(l, input, false);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 2.6f, result->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.7f, result->data[1]);

    tensor_free(input);
    tensor_free(result);
    layer_free(l);
}

void test_linear_zero_input(void) {
    Layer *l = layer_linear_create(2, 2, true);
    Tensor **params;
    size_t count;
    layer_parameters(l, &params, &count);
    Tensor *b = params[1];
    b->data[0] = 5.0f;
    b->data[1] = -5.0f;
    if (params)
        free(params);

    uint64_t in_shape[] = {1, 2};
    Tensor *input = tensor_zeros(in_shape, 2, false);

    Tensor *result = layer_forward(l, input, false);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 5.0f, result->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, -5.0f, result->data[1]);

    tensor_free(input);
    tensor_free(result);
    layer_free(l);
}

void test_linear_large_batch(void) {
    uint64_t in_features = 10;
    uint64_t out_features = 5;
    uint64_t batch_size = 128;

    Layer *l = layer_linear_create(in_features, out_features, true);

    uint64_t in_shape[] = {batch_size, in_features};
    Tensor *input = tensor_zeros(in_shape, 2, false);

    Tensor *out = layer_forward(l, input, false);

    TEST_ASSERT_EQUAL_UINT64(2, out->ndim);
    TEST_ASSERT_EQUAL_UINT64(batch_size, out->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(out_features, out->shape[1]);

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_linear_initialization_not_zero(void) {
    Layer *l = layer_linear_create(50, 50, false);
    Tensor **params;
    size_t count;
    layer_parameters(l, &params, &count);

    int nonzero = 0;
    for (uint64_t i = 0; i < params[0]->size; i++) {
        if (fabs(params[0]->data[i]) > 1e-9f) {
            nonzero++;
        }
    }

    TEST_ASSERT_TRUE(nonzero > 0);

    if (params)
        free(params);
    layer_free(l);
}

void test_dropout_identity_inference(void) {
    Layer *l = layer_dropout_create(0.5f);

    float32_t in_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint64_t in_shape[] = {4};
    Tensor *input = create_tensor_from_data(in_data, in_shape, 1);

    Tensor *out = layer_forward(l, input, false);

    for (int i = 0; i < 4; ++i) {
        TEST_ASSERT_EQUAL_FLOAT(in_data[i], out->data[i]);
    }

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_dropout_p0_training(void) {
    Layer *l = layer_dropout_create(0.0f);

    float32_t in_data[] = {1.0f, 2.0f};
    uint64_t in_shape[] = {2};
    Tensor *input = create_tensor_from_data(in_data, in_shape, 1);

    Tensor *out = layer_forward(l, input, true);

    TEST_ASSERT_EQUAL_FLOAT(1.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, out->data[1]);

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_dropout_p1_training(void) {
    Layer *l = layer_dropout_create(1.0f);

    float32_t in_data[] = {1.0f, 2.0f, 3.0f};
    uint64_t in_shape[] = {3};
    Tensor *input = create_tensor_from_data(in_data, in_shape, 1);

    Tensor *out = layer_forward(l, input, true);

    for (int i = 0; i < 3; ++i) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, out->data[i]);
    }

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_dropout_scaling_p05(void) {
    Layer *l = layer_dropout_create(0.5f);

    uint64_t size = 1000;
    uint64_t shape[] = {size};
    float32_t *data = (float32_t *)malloc(size * sizeof(float32_t));
    for (uint64_t i = 0; i < size; ++i)
        data[i] = 1.0f;

    Tensor *input = create_tensor_from_data(data, shape, 1);
    free(data);

    Tensor *out = layer_forward(l, input, true);

    int zeros = 0;
    int scaled = 0;

    for (uint64_t i = 0; i < size; ++i) {
        float32_t val = out->data[i];
        if (fabs(val) < 1e-6) {
            zeros++;
        } else if (fabs(val - 2.0f) < 1e-6) {
            scaled++;
        } else {
            TEST_FAIL();
        }
    }

    TEST_ASSERT_TRUE(zeros > 0);
    TEST_ASSERT_TRUE(scaled > 0);

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_dropout_scaling_p025(void) {
    float32_t p = 0.25f;
    Layer *l = layer_dropout_create(p);

    uint64_t size = 1000;
    uint64_t shape[] = {size};
    float32_t *data = (float32_t *)malloc(size * sizeof(float32_t));
    for (uint64_t i = 0; i < size; ++i)
        data[i] = 1.0f;

    Tensor *input = create_tensor_from_data(data, shape, 1);
    free(data);

    Tensor *out = layer_forward(l, input, true);

    float32_t expected_scale = 1.0f / (1.0f - p);

    for (uint64_t i = 0; i < size; ++i) {
        float32_t val = out->data[i];
        if (fabs(val) > 1e-6) {
            TEST_ASSERT_FLOAT_WITHIN(1e-5, expected_scale, val);
        }
    }

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_dropout_scaling_p075(void) {
    float32_t p = 0.75f;
    Layer *l = layer_dropout_create(p);

    uint64_t size = 1000;
    uint64_t shape[] = {size};
    float32_t *data = (float32_t *)malloc(size * sizeof(float32_t));
    for (uint64_t i = 0; i < size; ++i)
        data[i] = 1.0f;

    Tensor *input = create_tensor_from_data(data, shape, 1);
    free(data);

    Tensor *out = layer_forward(l, input, true);

    float32_t expected_scale = 1.0f / (1.0f - p);

    for (uint64_t i = 0; i < size; ++i) {
        float32_t val = out->data[i];
        if (fabs(val) > 1e-6) {
            TEST_ASSERT_FLOAT_WITHIN(1e-5, expected_scale, val);
        }
    }

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_sequential_chaining(void) {
    Layer *l1 = layer_linear_create(2, 2, false);
    Layer *l2 = layer_dropout_create(0.0f);
    Layer *l3 = layer_linear_create(2, 1, false);

    Layer **layers_arr = (Layer **)malloc(3 * sizeof(Layer *));
    layers_arr[0] = l1;
    layers_arr[1] = l2;
    layers_arr[2] = l3;

    Layer *seq = layer_sequential_create(layers_arr, 3);
    free(layers_arr);

    Tensor **params;
    size_t count;
    layer_parameters(seq, &params, &count);
    TEST_ASSERT_EQUAL_UINT64(2, count);
    if (params)
        free(params);

    Tensor **p1;
    size_t c1;
    layer_parameters(l1, &p1, &c1);
    Tensor *w1 = p1[0];
    w1->data[0] = 1.0f;
    w1->data[1] = 0.0f;
    w1->data[2] = 0.0f;
    w1->data[3] = 1.0f;
    if (p1)
        free(p1);

    Tensor **p3;
    size_t c3;
    layer_parameters(l3, &p3, &c3);
    Tensor *w3 = p3[0];
    w3->data[0] = 1.0f;
    w3->data[1] = 1.0f;
    if (p3)
        free(p3);

    float32_t in_data[] = {1.0f, 2.0f};
    uint64_t in_shape[] = {1, 2};
    Tensor *input = create_tensor_from_data(in_data, in_shape, 2);

    Tensor *out = layer_forward(seq, input, true);

    TEST_ASSERT_EQUAL_FLOAT(3.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    layer_free(seq);
}

void test_sequential_empty(void) {
    Layer *seq = layer_sequential_create(NULL, 0);

    float32_t in_data[] = {42.0f};
    uint64_t in_shape[] = {1};
    Tensor *input = create_tensor_from_data(in_data, in_shape, 1);

    Tensor *out = layer_forward(seq, input, false);

    TEST_ASSERT_EQUAL_FLOAT(42.0f, out->data[0]);
    TEST_ASSERT_NOT_EQUAL(input, out);

    tensor_free(input);
    tensor_free(out);
    layer_free(seq);
}

void test_sequential_nested(void) {
    Layer *l1 = layer_linear_create(1, 1, false);
    Tensor **p1;
    size_t c1;
    layer_parameters(l1, &p1, &c1);
    p1[0]->data[0] = 1.0f;
    if (p1)
        free(p1);

    Layer **arr1 = (Layer **)malloc(sizeof(Layer *));
    arr1[0] = l1;
    Layer *seq1 = layer_sequential_create(arr1, 1);
    free(arr1);

    Layer *l2 = layer_linear_create(1, 1, false);
    Tensor **p2;
    size_t c2;
    layer_parameters(l2, &p2, &c2);
    p2[0]->data[0] = 2.0f;
    if (p2)
        free(p2);

    Layer **arr2 = (Layer **)malloc(2 * sizeof(Layer *));
    arr2[0] = seq1;
    arr2[1] = l2;
    Layer *seq2 = layer_sequential_create(arr2, 2);
    free(arr2);

    float32_t in_data[] = {10.0f};
    uint64_t in_shape[] = {1, 1};
    Tensor *input = create_tensor_from_data(in_data, in_shape, 2);

    Tensor *out = layer_forward(seq2, input, false);

    TEST_ASSERT_EQUAL_FLOAT(20.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    layer_free(seq2);
}

void test_linear_mixed_signs(void) {
    Layer *l = layer_linear_create(2, 2, true);
    Tensor **params;
    size_t count;
    layer_parameters(l, &params, &count);

    params[0]->data[0] = 1.0f;
    params[0]->data[1] = -1.0f;
    params[0]->data[2] = -1.0f;
    params[0]->data[3] = 1.0f;

    params[1]->data[0] = 0.5f;
    params[1]->data[1] = -0.5f;

    if (params)
        free(params);

    float32_t in_data[] = {-2.0f, 2.0f};
    uint64_t in_shape[] = {1, 2};
    Tensor *input = create_tensor_from_data(in_data, in_shape, 2);

    Tensor *out = layer_forward(l, input, false);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, -3.5f, out->data[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.5f, out->data[1]);

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_dropout_p09_training(void) {
    float32_t p = 0.9f;
    Layer *l = layer_dropout_create(p);

    uint64_t size = 2000;
    uint64_t shape[] = {size};
    float32_t *data = (float32_t *)malloc(size * sizeof(float32_t));
    for (uint64_t i = 0; i < size; ++i)
        data[i] = 1.0f;

    Tensor *input = create_tensor_from_data(data, shape, 1);
    free(data);

    Tensor *out = layer_forward(l, input, true);

    int zeros = 0;
    for (uint64_t i = 0; i < size; ++i) {
        if (out->data[i] == 0.0f)
            zeros++;
    }

    float32_t zero_ratio = (float32_t)zeros / (float32_t)size;
    TEST_ASSERT_FLOAT_WITHIN(0.05f, 0.9f, zero_ratio);

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_sequential_deep(void) {
    size_t depth = 5;
    Layer **layers = (Layer **)malloc(depth * sizeof(Layer *));
    for (size_t i = 0; i < depth; ++i) {
        layers[i] = layer_linear_create(2, 2, false);
        Tensor **p;
        size_t c;
        layer_parameters(layers[i], &p, &c);
        p[0]->data[0] = 1.0f;
        p[0]->data[1] = 0.0f;
        p[0]->data[2] = 0.0f;
        p[0]->data[3] = 1.0f;
        if (p)
            free(p);
    }

    Layer *seq = layer_sequential_create(layers, depth);
    free(layers);

    float32_t in_data[] = {5.0f, -3.0f};
    uint64_t in_shape[] = {1, 2};
    Tensor *input = create_tensor_from_data(in_data, in_shape, 2);

    Tensor *out = layer_forward(seq, input, false);

    TEST_ASSERT_EQUAL_FLOAT(5.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(-3.0f, out->data[1]);

    tensor_free(input);
    tensor_free(out);
    layer_free(seq);
}

void test_sequential_linear_dropout_linear(void) {
    Layer *l1 = layer_linear_create(2, 4, false);
    Layer *l2 = layer_dropout_create(0.5f);
    Layer *l3 = layer_linear_create(4, 1, false);

    Layer **arr = (Layer **)malloc(3 * sizeof(Layer *));
    arr[0] = l1;
    arr[1] = l2;
    arr[2] = l3;

    Layer *seq = layer_sequential_create(arr, 3);
    free(arr);

    Tensor **p;
    size_t c;
    layer_parameters(seq, &p, &c);

    for (int i = 0; i < 8; ++i)
        p[0]->data[i] = 1.0f;
    for (int i = 0; i < 4; ++i)
        p[1]->data[i] = 0.5f;

    if (p)
        free(p);

    float32_t in_data[] = {1.0f, 1.0f};
    uint64_t in_shape[] = {1, 2};
    Tensor *input = create_tensor_from_data(in_data, in_shape, 2);

    Tensor *out_inf = layer_forward(seq, input, false);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, out_inf->data[0]);
    tensor_free(out_inf);

    Tensor *out_train = layer_forward(seq, input, true);
    TEST_ASSERT_NOT_NULL(out_train);
    tensor_free(out_train);

    tensor_free(input);
    layer_free(seq);
}

void test_linear_bias_init_zero(void) {
    Layer *l = layer_linear_create(10, 5, true);
    Tensor **params;
    size_t count;
    layer_parameters(l, &params, &count);
    Tensor *b = params[1];
    for (uint64_t i = 0; i < b->size; ++i) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, b->data[i]);
    }
    if (params)
        free(params);
    layer_free(l);
}

void test_linear_input_requires_grad(void) {
    Layer *l = layer_linear_create(2, 2, false);
    uint64_t shape[] = {1, 2};
    Tensor *input = tensor_zeros(shape, 2, true);
    Tensor *out = layer_forward(l, input, false);
    TEST_ASSERT_TRUE(out->requires_grad);
    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_linear_input_no_grad(void) {
    Layer *l = layer_linear_create(2, 2, false);
    uint64_t shape[] = {1, 2};
    Tensor *input = tensor_zeros(shape, 2, false);
    Tensor *out = layer_forward(l, input, false);
    TEST_ASSERT_TRUE(out->requires_grad);
    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_linear_scalar(void) {
    Layer *l = layer_linear_create(1, 1, false);
    Tensor **params;
    size_t c;
    layer_parameters(l, &params, &c);
    params[0]->data[0] = 2.0f;
    if (params)
        free(params);

    float32_t val[] = {3.0f};
    uint64_t shape[] = {1, 1};
    Tensor *input = create_tensor_from_data(val, shape, 2);
    Tensor *out = layer_forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(6.0f, out->data[0]);
    TEST_ASSERT_EQUAL_UINT64(2, out->ndim);

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_linear_vector_to_scalar(void) {
    Layer *l = layer_linear_create(3, 1, false);
    Tensor **params;
    size_t c;
    layer_parameters(l, &params, &c);
    params[0]->data[0] = 1.0f;
    params[0]->data[1] = 1.0f;
    params[0]->data[2] = 1.0f;
    if (params)
        free(params);

    float32_t val[] = {1.0f, 2.0f, 3.0f};
    uint64_t shape[] = {1, 3};
    Tensor *input = create_tensor_from_data(val, shape, 2);
    Tensor *out = layer_forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(6.0f, out->data[0]);
    TEST_ASSERT_EQUAL_UINT64(1, out->shape[1]);

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_linear_scalar_to_vector(void) {
    Layer *l = layer_linear_create(1, 3, false);
    Tensor **params;
    size_t c;
    layer_parameters(l, &params, &c);
    params[0]->data[0] = 1.0f;
    params[0]->data[1] = 2.0f;
    params[0]->data[2] = 3.0f;
    if (params)
        free(params);

    float32_t val[] = {2.0f};
    uint64_t shape[] = {1, 1};
    Tensor *input = create_tensor_from_data(val, shape, 2);
    Tensor *out = layer_forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(2.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(4.0f, out->data[1]);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, out->data[2]);

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_linear_nan_input(void) {
    Layer *l = layer_linear_create(2, 2, false);
    float32_t val[] = {NAN, 1.0f};
    uint64_t shape[] = {1, 2};
    Tensor *input = create_tensor_from_data(val, shape, 2);
    Tensor *out = layer_forward(l, input, false);

    TEST_ASSERT_TRUE(isnan(out->data[0]) || isnan(out->data[1]));

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_linear_inf_input(void) {
    Layer *l = layer_linear_create(1, 1, false);
    Tensor **params;
    size_t c;
    layer_parameters(l, &params, &c);
    params[0]->data[0] = 1.0f;
    if (params)
        free(params);

    float32_t val[] = {INFINITY};
    uint64_t shape[] = {1, 1};
    Tensor *input = create_tensor_from_data(val, shape, 2);
    Tensor *out = layer_forward(l, input, false);

    TEST_ASSERT_TRUE(isinf(out->data[0]));

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_linear_reuse(void) {
    Layer *l = layer_linear_create(2, 2, false);
    Tensor **params;
    size_t c;
    layer_parameters(l, &params, &c);
    params[0]->data[0] = 1;
    params[0]->data[1] = 0;
    params[0]->data[2] = 0;
    params[0]->data[3] = 1;
    if (params)
        free(params);

    float32_t val[] = {1.0f, 2.0f};
    uint64_t shape[] = {1, 2};
    Tensor *input = create_tensor_from_data(val, shape, 2);

    for (int i = 0; i < 3; i++) {
        Tensor *out = layer_forward(l, input, false);
        TEST_ASSERT_EQUAL_FLOAT(1.0f, out->data[0]);
        TEST_ASSERT_EQUAL_FLOAT(2.0f, out->data[1]);
        tensor_free(out);
    }

    tensor_free(input);
    layer_free(l);
}

void test_linear_negative_input(void) {
    Layer *l = layer_linear_create(1, 1, false);
    Tensor **params;
    size_t c;
    layer_parameters(l, &params, &c);
    params[0]->data[0] = -1.0f;
    if (params)
        free(params);

    float32_t val[] = {-5.0f};
    uint64_t shape[] = {1, 1};
    Tensor *input = create_tensor_from_data(val, shape, 2);
    Tensor *out = layer_forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(5.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_dropout_inference_p01(void) {
    Layer *l = layer_dropout_create(0.1f);
    float32_t val[] = {1.0f, 2.0f, 3.0f};
    uint64_t shape[] = {1, 3};
    Tensor *input = create_tensor_from_data(val, shape, 2);
    Tensor *out = layer_forward(l, input, false);
    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_EQUAL_FLOAT(val[i], out->data[i]);
    }
    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_dropout_randomness(void) {
    Layer *l = layer_dropout_create(0.5f);
    uint64_t size = 100;
    uint64_t shape[] = {size};
    float32_t *data = (float32_t *)malloc(size * sizeof(float32_t));
    for (uint64_t i = 0; i < size; i++)
        data[i] = 1.0f;
    Tensor *input = create_tensor_from_data(data, shape, 1);
    free(data);

    Tensor *out1 = layer_forward(l, input, true);
    Tensor *out2 = layer_forward(l, input, true);

    int diff_count = 0;
    for (uint64_t i = 0; i < size; i++) {
        if (out1->data[i] != out2->data[i])
            diff_count++;
    }
    TEST_ASSERT_TRUE(diff_count > 0);

    tensor_free(input);
    tensor_free(out1);
    tensor_free(out2);
    layer_free(l);
}

void test_dropout_output_shape(void) {
    Layer *l = layer_dropout_create(0.5f);
    uint64_t shape[] = {2, 3, 4};
    Tensor *input = tensor_zeros(shape, 3, false);
    Tensor *out = layer_forward(l, input, true);

    TEST_ASSERT_EQUAL_UINT64(3, out->ndim);
    TEST_ASSERT_EQUAL_UINT64(2, out->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(3, out->shape[1]);
    TEST_ASSERT_EQUAL_UINT64(4, out->shape[2]);

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_dropout_requires_grad_propagation(void) {
    Layer *l = layer_dropout_create(0.5f);
    uint64_t shape[] = {1};
    Tensor *input = tensor_zeros(shape, 1, true);
    Tensor *out = layer_forward(l, input, true);
    TEST_ASSERT_TRUE(out->requires_grad);
    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_dropout_all_zeros(void) {
    Layer *l = layer_dropout_create(0.5f);
    uint64_t shape[] = {10};
    Tensor *input = tensor_zeros(shape, 1, false);
    Tensor *out = layer_forward(l, input, true);
    for (int i = 0; i < 10; i++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, out->data[i]);
    }
    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_dropout_nan_propagation(void) {
    Layer *l = layer_dropout_create(0.0f);
    float32_t val[] = {NAN};
    uint64_t shape[] = {1};
    Tensor *input = create_tensor_from_data(val, shape, 1);
    Tensor *out = layer_forward(l, input, true);
    TEST_ASSERT_TRUE(isnan(out->data[0]));
    tensor_free(input);
    tensor_free(out);
    layer_free(l);

    l = layer_dropout_create(0.5f);
    input = create_tensor_from_data(val, shape, 1);
    out = layer_forward(l, input, true);
    TEST_ASSERT_TRUE(isnan(out->data[0]));
    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_sequential_single_layer(void) {
    Layer *l1 = layer_linear_create(2, 2, false);
    Layer **arr = (Layer **)malloc(sizeof(Layer *));
    arr[0] = l1;
    Layer *seq = layer_sequential_create(arr, 1);
    free(arr);

    float32_t val[] = {1.0f, 2.0f};
    uint64_t shape[] = {1, 2};
    Tensor *input = create_tensor_from_data(val, shape, 2);

    Tensor **params;
    size_t c;
    layer_parameters(l1, &params, &c);
    params[0]->data[0] = 1;
    params[0]->data[1] = 0;
    params[0]->data[2] = 0;
    params[0]->data[3] = 1;
    if (params)
        free(params);

    Tensor *out = layer_forward(seq, input, false);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, out->data[0]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, out->data[1]);

    tensor_free(input);
    tensor_free(out);
    layer_free(seq);
}

void test_sequential_param_count_correct(void) {
    Layer *l1 = layer_linear_create(2, 2, true);
    Layer *l2 = layer_linear_create(2, 2, false);
    Layer **arr = (Layer **)malloc(2 * sizeof(Layer *));
    arr[0] = l1;
    arr[1] = l2;
    Layer *seq = layer_sequential_create(arr, 2);
    free(arr);

    Tensor **params;
    size_t count;
    layer_parameters(seq, &params, &count);
    TEST_ASSERT_EQUAL_UINT64(3, count);
    if (params)
        free(params);
    layer_free(seq);
}

void test_sequential_with_dropout_only(void) {
    Layer *l1 = layer_dropout_create(0.5f);
    Layer *l2 = layer_dropout_create(0.5f);
    Layer **arr = (Layer **)malloc(2 * sizeof(Layer *));
    arr[0] = l1;
    arr[1] = l2;
    Layer *seq = layer_sequential_create(arr, 2);
    free(arr);

    uint64_t shape[] = {100};
    Tensor *input = tensor_zeros(shape, 1, false);
    Tensor *out = layer_forward(seq, input, true);
    TEST_ASSERT_EQUAL_UINT64(1, out->ndim);
    TEST_ASSERT_EQUAL_UINT64(100, out->shape[0]);

    tensor_free(input);
    tensor_free(out);
    layer_free(seq);
}

void test_sequential_training_mode_propagation(void) {
    Layer *l1 = layer_dropout_create(1.0f);
    Layer **arr = (Layer **)malloc(sizeof(Layer *));
    arr[0] = l1;
    Layer *seq = layer_sequential_create(arr, 1);
    free(arr);

    float32_t val[] = {1.0f};
    uint64_t shape[] = {1};
    Tensor *input = create_tensor_from_data(val, shape, 1);

    Tensor *out = layer_forward(seq, input, true);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    layer_free(seq);
}

void test_sequential_inference_mode_propagation(void) {
    Layer *l1 = layer_dropout_create(1.0f);
    Layer **arr = (Layer **)malloc(sizeof(Layer *));
    arr[0] = l1;
    Layer *seq = layer_sequential_create(arr, 1);
    free(arr);

    float32_t val[] = {5.0f};
    uint64_t shape[] = {1};
    Tensor *input = create_tensor_from_data(val, shape, 1);

    Tensor *out = layer_forward(seq, input, false);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    layer_free(seq);
}

void test_layer_free_null(void) {
    layer_free(NULL);
    TEST_ASSERT_TRUE(true);
}

void test_sequential_param_pointers_distinct(void) {
    Layer *l1 = layer_linear_create(1, 1, false);
    Layer *l2 = layer_linear_create(1, 1, false);
    Layer **arr = (Layer **)malloc(2 * sizeof(Layer *));
    arr[0] = l1;
    arr[1] = l2;
    Layer *seq = layer_sequential_create(arr, 2);
    free(arr);

    Tensor **params;
    size_t count;
    layer_parameters(seq, &params, &count);

    TEST_ASSERT_EQUAL_UINT64(2, count);
    if (count >= 2) {
        TEST_ASSERT_NOT_EQUAL(params[0], params[1]);
    }

    if (params)
        free(params);
    layer_free(seq);
}

void test_sequential_shape_consistency(void) {
    Layer *l1 = layer_linear_create(10, 5, false);
    Layer *l2 = layer_linear_create(5, 2, false);
    Layer **arr = (Layer **)malloc(2 * sizeof(Layer *));
    arr[0] = l1;
    arr[1] = l2;
    Layer *seq = layer_sequential_create(arr, 2);
    free(arr);

    uint64_t shape[] = {4, 10};
    Tensor *input = tensor_zeros(shape, 2, false);
    Tensor *out = layer_forward(seq, input, false);

    TEST_ASSERT_EQUAL_UINT64(2, out->ndim);
    TEST_ASSERT_EQUAL_UINT64(4, out->shape[0]);
    TEST_ASSERT_EQUAL_UINT64(2, out->shape[1]);

    tensor_free(input);
    tensor_free(out);
    layer_free(seq);
}

void test_sequential_reuse(void) {
    Layer *l1 = layer_linear_create(1, 1, false);
    Layer **arr = (Layer **)malloc(sizeof(Layer *));
    arr[0] = l1;
    Layer *seq = layer_sequential_create(arr, 1);
    free(arr);

    float32_t val[] = {1.0f};
    uint64_t shape[] = {1, 1};
    Tensor *input = create_tensor_from_data(val, shape, 2);

    for (int i = 0; i < 3; i++) {
        Tensor *out = layer_forward(seq, input, false);
        TEST_ASSERT_NOT_NULL(out);
        tensor_free(out);
    }
    tensor_free(input);
    layer_free(seq);
}

void test_sequential_large_chain(void) {
    size_t count = 20;
    Layer **arr = (Layer **)malloc(count * sizeof(Layer *));
    for (size_t i = 0; i < count; i++) {
        arr[i] = layer_linear_create(1, 1, false);
        Tensor **p;
        size_t c;
        layer_parameters(arr[i], &p, &c);
        p[0]->data[0] = 1.0f;
        if (p)
            free(p);
    }
    Layer *seq = layer_sequential_create(arr, count);
    free(arr);

    float32_t val[] = {7.0f};
    uint64_t shape[] = {1, 1};
    Tensor *input = create_tensor_from_data(val, shape, 2);
    Tensor *out = layer_forward(seq, input, false);

    TEST_ASSERT_EQUAL_FLOAT(7.0f, out->data[0]);

    tensor_free(input);
    tensor_free(out);
    layer_free(seq);
}

void test_dropout_p05_stats(void) {
    Layer *l = layer_dropout_create(0.5f);
    uint64_t size = 1000;
    uint64_t shape[] = {size};
    Tensor *input = tensor_zeros(shape, 1, false);
    for (uint64_t i = 0; i < size; i++) {
        input->data[i] = 1.0f;
    }

    Tensor *out = layer_forward(l, input, true);
    int zero_count = 0;
    for (uint64_t i = 0; i < size; i++) {
        if (out->data[i] == 0.0f)
            zero_count++;
    }
    TEST_ASSERT_TRUE(zero_count > 400 && zero_count < 600);

    tensor_free(input);
    tensor_free(out);
    layer_free(l);
}

void test_linear_all_ones_weights(void) {
    Layer *l = layer_linear_create(2, 1, false);
    Tensor **p;
    size_t c;
    layer_parameters(l, &p, &c);
    p[0]->data[0] = 1.0f;
    p[0]->data[1] = 1.0f;
    if (p)
        free(p);

    float32_t val[] = {2.0f, 3.0f};
    uint64_t shape[] = {1, 2};
    Tensor *input = create_tensor_from_data(val, shape, 2);
    Tensor *out = layer_forward(l, input, false);

    TEST_ASSERT_EQUAL_FLOAT(5.0f, out->data[0]);

    tensor_free(out);
    tensor_free(input);
    layer_free(l);
}

void test_sequential_linear_dropout(void) {
    Layer *l1 = layer_linear_create(1, 1, false);
    Layer *l2 = layer_dropout_create(0.0f);
    Layer **arr = (Layer **)malloc(2 * sizeof(Layer *));
    arr[0] = l1;
    arr[1] = l2;
    Layer *seq = layer_sequential_create(arr, 2);
    free(arr);

    Tensor **p;
    size_t c;
    layer_parameters(l1, &p, &c);
    p[0]->data[0] = 1.0f;
    if (p)
        free(p);

    float32_t val[] = {5.0f};
    uint64_t shape[] = {1, 1};
    Tensor *input = create_tensor_from_data(val, shape, 2);
    Tensor *out = layer_forward(seq, input, true);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, out->data[0]);
    tensor_free(input);
    tensor_free(out);
    layer_free(seq);
}

void test_sequential_dropout_linear(void) {
    Layer *l1 = layer_dropout_create(0.0f);
    Layer *l2 = layer_linear_create(1, 1, false);
    Layer **arr = (Layer **)malloc(2 * sizeof(Layer *));
    arr[0] = l1;
    arr[1] = l2;
    Layer *seq = layer_sequential_create(arr, 2);
    free(arr);

    Tensor **p;
    size_t c;
    layer_parameters(l2, &p, &c);
    p[0]->data[0] = 1.0f;
    if (p)
        free(p);

    float32_t val[] = {9.0f};
    uint64_t shape[] = {1, 1};
    Tensor *input = create_tensor_from_data(val, shape, 2);
    Tensor *out = layer_forward(seq, input, true);
    TEST_ASSERT_EQUAL_FLOAT(9.0f, out->data[0]);
    tensor_free(input);
    tensor_free(out);
    layer_free(seq);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_linear_creation);
    RUN_TEST(test_linear_creation_no_bias);
    RUN_TEST(test_linear_forward_shape);
    RUN_TEST(test_linear_forward_values);
    RUN_TEST(test_linear_zero_input);
    RUN_TEST(test_linear_large_batch);
    RUN_TEST(test_linear_initialization_not_zero);
    RUN_TEST(test_linear_mixed_signs);
    RUN_TEST(test_dropout_identity_inference);
    RUN_TEST(test_dropout_p0_training);
    RUN_TEST(test_dropout_p1_training);
    RUN_TEST(test_dropout_scaling_p05);
    RUN_TEST(test_dropout_scaling_p025);
    RUN_TEST(test_dropout_scaling_p075);
    RUN_TEST(test_dropout_p09_training);
    RUN_TEST(test_sequential_chaining);
    RUN_TEST(test_sequential_empty);
    RUN_TEST(test_sequential_nested);
    RUN_TEST(test_sequential_deep);
    RUN_TEST(test_sequential_linear_dropout_linear);
    RUN_TEST(test_linear_bias_init_zero);
    RUN_TEST(test_linear_input_requires_grad);
    RUN_TEST(test_linear_input_no_grad);
    RUN_TEST(test_linear_scalar);
    RUN_TEST(test_linear_vector_to_scalar);
    RUN_TEST(test_linear_scalar_to_vector);
    RUN_TEST(test_linear_nan_input);
    RUN_TEST(test_linear_inf_input);
    RUN_TEST(test_linear_reuse);
    RUN_TEST(test_linear_negative_input);
    RUN_TEST(test_dropout_inference_p01);
    RUN_TEST(test_dropout_randomness);
    RUN_TEST(test_dropout_output_shape);
    RUN_TEST(test_dropout_requires_grad_propagation);
    RUN_TEST(test_dropout_all_zeros);
    RUN_TEST(test_dropout_nan_propagation);
    RUN_TEST(test_sequential_single_layer);
    RUN_TEST(test_sequential_param_count_correct);
    RUN_TEST(test_sequential_with_dropout_only);
    RUN_TEST(test_sequential_training_mode_propagation);
    RUN_TEST(test_sequential_inference_mode_propagation);
    RUN_TEST(test_layer_free_null);
    RUN_TEST(test_sequential_param_pointers_distinct);
    RUN_TEST(test_sequential_shape_consistency);
    RUN_TEST(test_sequential_reuse);
    RUN_TEST(test_sequential_large_chain);
    RUN_TEST(test_dropout_p05_stats);
    RUN_TEST(test_linear_all_ones_weights);
    RUN_TEST(test_sequential_linear_dropout);
    RUN_TEST(test_sequential_dropout_linear);
    return UNITY_END();
}
