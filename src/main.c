#include "autograd.h"
#include "ops/activations.h"
#include "ops/arithmetic.h"
#include "ops/convolutions.h"
#include "ops/losses.h"
#include "ops/reshapes.h"
#include "optimizers.h"
#include "tensor.h"
#include "utils/cifar10.h"
#include "utils/metrics.h"
#include "utils/tqdm.h"
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BATCH_SIZE 64
#define LEARNING_RATE 0.001f
#define WEIGHT_DECAY 0.0005f
#define NUM_EPOCHS 15
#define DROPOUT_RATE 0.3f

typedef struct {
    // conv layer 1: 3 -> 32 channels
    Tensor *conv1_w;
    Tensor *conv1_b;

    // conv layer 2: 32 -> 64 channels
    Tensor *conv2_w;
    Tensor *conv2_b;

    // conv layer 3: 64 -> 128 channels
    Tensor *conv3_w;
    Tensor *conv3_b;

    // fully connected layer 1: 2048 -> 256
    Tensor *fc1_w;
    Tensor *fc1_b;

    // fully connected layer 2: 256 -> 10
    Tensor *fc2_w;
    Tensor *fc2_b;
} Model;

Tensor *create_conv_weights(uint64_t out_channels, uint64_t in_channels, uint64_t kernel_size, bool requires_grad) {
    const uint64_t shape[] = {out_channels, in_channels, kernel_size, kernel_size};
    size_t total = out_channels * in_channels * kernel_size * kernel_size;
    float32_t *data = (float32_t *)malloc(total * sizeof(float32_t));
    assert(data != NULL && "malloc failed");

    float32_t fan_in = (float32_t)(in_channels * kernel_size * kernel_size);
    float32_t limit = sqrtf(2.0f / fan_in);

    for (size_t i = 0; i < total; i++) {
        float32_t r = (float32_t)rand() / (float32_t)RAND_MAX;
        data[i] = (r * 2.0f * limit) - limit;
    }

    Tensor *w = tensor_create(data, shape, 4, requires_grad);
    free(data);
    return w;
}

Tensor *create_conv_bias(uint64_t out_channels, bool requires_grad) {
    const uint64_t shape[] = {out_channels};
    return tensor_zeros(shape, 1, requires_grad);
}

Tensor *create_fc_weights(uint64_t in_features, uint64_t out_features, bool requires_grad) {
    const uint64_t shape[] = {in_features, out_features};
    size_t total = in_features * out_features;
    float32_t *data = (float32_t *)malloc(total * sizeof(float32_t));
    assert(data != NULL && "malloc failed");

    float32_t limit = 1.0f / sqrtf((float32_t)in_features);

    for (size_t i = 0; i < total; i++) {
        float32_t r = (float32_t)rand() / (float32_t)RAND_MAX;
        data[i] = (r * 2.0f * limit) - limit;
    }

    Tensor *w = tensor_create(data, shape, 2, requires_grad);
    free(data);
    return w;
}

Tensor *create_fc_bias(uint64_t out_features, bool requires_grad) {
    const uint64_t shape[] = {out_features};
    return tensor_zeros(shape, 1, requires_grad);
}

Model *create_model(void) {
    Model *model = (Model *)calloc(1, sizeof(Model));
    assert(model != NULL && "calloc failed");

    // conv1: 3 -> 32 channels, 3x3 kernel
    model->conv1_w = create_conv_weights(32, 3, 3, true);
    model->conv1_b = create_conv_bias(32, true);

    // conv2: 32 -> 64 channels, 3x3 kernel
    model->conv2_w = create_conv_weights(64, 32, 3, true);
    model->conv2_b = create_conv_bias(64, true);

    // conv3: 64 -> 128 channels, 3x3 kernel
    model->conv3_w = create_conv_weights(128, 64, 3, true);
    model->conv3_b = create_conv_bias(128, true);

    // fc1: 2048 -> 256
    model->fc1_w = create_fc_weights(2048, 256, true);
    model->fc1_b = create_fc_bias(256, true);

    // fc2: 256 -> 10
    model->fc2_w = create_fc_weights(256, 10, true);
    model->fc2_b = create_fc_bias(10, true);

    return model;
}

void get_all_parameters(Model *model, Tensor ***out_params, size_t *out_count) {
    assert(model != NULL);
    assert(out_params != NULL);
    assert(out_count != NULL);

    // total: 6 conv params + 4 FC params = 10
    size_t total_count = 10;
    Tensor **params = (Tensor **)malloc(total_count * sizeof(Tensor *));
    assert(params != NULL && "malloc failed");

    params[0] = model->conv1_w;
    params[1] = model->conv1_b;
    params[2] = model->conv2_w;
    params[3] = model->conv2_b;
    params[4] = model->conv3_w;
    params[5] = model->conv3_b;
    params[6] = model->fc1_w;
    params[7] = model->fc1_b;
    params[8] = model->fc2_w;
    params[9] = model->fc2_b;

    *out_params = params;
    *out_count = total_count;
}

void free_model(Model *model) {
    if (model == NULL) {
        return;
    }

    if (model->conv1_w != NULL)
        tensor_free(model->conv1_w);
    if (model->conv1_b != NULL)
        tensor_free(model->conv1_b);
    if (model->conv2_w != NULL)
        tensor_free(model->conv2_w);
    if (model->conv2_b != NULL)
        tensor_free(model->conv2_b);
    if (model->conv3_w != NULL)
        tensor_free(model->conv3_w);
    if (model->conv3_b != NULL)
        tensor_free(model->conv3_b);
    if (model->fc1_w != NULL)
        tensor_free(model->fc1_w);
    if (model->fc1_b != NULL)
        tensor_free(model->fc1_b);
    if (model->fc2_w != NULL)
        tensor_free(model->fc2_w);
    if (model->fc2_b != NULL)
        tensor_free(model->fc2_b);

    free(model);
}

Tensor *forward_cnn(const Model *model, const Tensor *x, bool training) {
    assert(model != NULL);
    assert(x != NULL);
    (void)training; // unused for now (no dropout)

    // conv1: (batch, 3, 32, 32) -> (batch, 32, 32, 32) -> ReLU -> (batch, 32, 16, 16)
    const Tensor *conv1 = tensor_conv2d(x, model->conv1_w, model->conv1_b, 1, 1, 1);
    const Tensor *relu1 = tensor_relu(conv1);
    const Tensor *pool1 = tensor_maxpool2d(relu1, 2, 2, 0);

    // conv2: (batch, 32, 16, 16) -> (batch, 64, 16, 16) -> ReLU -> (batch, 64, 8, 8)
    const Tensor *conv2 = tensor_conv2d(pool1, model->conv2_w, model->conv2_b, 1, 1, 1);
    const Tensor *relu2 = tensor_relu(conv2);
    const Tensor *pool2 = tensor_maxpool2d(relu2, 2, 2, 0);

    // conv3: (batch, 64, 8, 8) -> (batch, 128, 8, 8) -> ReLU -> (batch, 128, 4, 4)
    const Tensor *conv3 = tensor_conv2d(pool2, model->conv3_w, model->conv3_b, 1, 1, 1);
    const Tensor *relu3 = tensor_relu(conv3);
    const Tensor *pool3 = tensor_maxpool2d(relu3, 2, 2, 0);

    // flatten: (batch, 128, 4, 4) -> (batch, 2048)
    const int64_t flat_shape[] = {(int64_t)pool3->shape[0], 2048};
    const Tensor *flat = tensor_reshape(pool3, flat_shape, 2);

    // fc1: (batch, 2048) @ (2048, 256) + (256,) -> (batch, 256)
    const Tensor *fc1 = tensor_matmul(flat, model->fc1_w);
    const Tensor *fc1_bias = tensor_add(fc1, model->fc1_b);
    const Tensor *fc1_relu = tensor_relu(fc1_bias);

    // fc2: (batch, 256) @ (256, 10) + (10,) -> (batch, 10)
    const Tensor *fc2 = tensor_matmul(fc1_relu, model->fc2_w);
    Tensor *logits = tensor_add(fc2, model->fc2_b);

    // don't free, we need them for backprop
    return logits;
}

void train_epoch(const Model *model, Optimizer *opt, const Tensor *train_images, const Tensor *train_labels, uint64_t batch_size, uint64_t epoch) {
    assert(model != NULL);
    assert(opt != NULL);
    assert(train_images != NULL);
    assert(train_labels != NULL);

    uint64_t num_batches = (NUM_TRAIN_SAMPLES + batch_size - 1) / batch_size;

    float32_t total_loss = 0.0f;
    float32_t total_acc = 0.0f;
    uint64_t processed_batches = 0;

    char prefix[64];
    char postfix[128];
    snprintf(prefix, sizeof(prefix), "epoch %2" PRIu64 "/%d", epoch, NUM_EPOCHS);

    for (uint64_t i = 0; i < num_batches; i++) {
        Tensor *batch_x = get_batch(train_images, i, batch_size);
        Tensor *batch_y = get_batch(train_labels, i, batch_size);

        if (batch_x == NULL || batch_y == NULL) {
            if (batch_x != NULL)
                tensor_free(batch_x);
            if (batch_y != NULL)
                tensor_free(batch_y);
            break;
        }

        // forward pass
        Tensor *logits = forward_cnn(model, batch_x, true);
        Tensor *loss = cross_entropy_loss(logits, batch_y);

        float32_t acc = accuracy(logits, batch_y);

        // backward pass
        optimizer_zero_grad(opt);
        backward(loss);
        optimizer_step(opt);

        total_loss += loss->data[0];
        total_acc += acc;
        processed_batches++;

        snprintf(postfix, sizeof(postfix), "loss=%.4f, acc=%.1f%%", loss->data[0], acc * 100.0f);
        tqdm(i + 1, num_batches, prefix, postfix);

        // cleanup
        tensor_free(batch_x);
        tensor_free(batch_y);
        tensor_free(logits);
        tensor_free(loss);
        arena_free(); // Critical: free autograd arena
    }

    float32_t avg_loss = total_loss / (float32_t)processed_batches;
    float32_t avg_acc = total_acc / (float32_t)processed_batches;
    printf("\tavg loss: %.4f, avg acc: %.2f%%\n", avg_loss, avg_acc * 100.0f);
}

float32_t evaluate(const Model *model, const Tensor *test_images, const Tensor *test_labels, uint64_t batch_size) {
    assert(model != NULL);
    assert(test_images != NULL);
    assert(test_labels != NULL);

    uint64_t num_batches = (NUM_TEST_SAMPLES + batch_size - 1) / batch_size;

    float32_t total_acc = 0.0f;
    uint64_t total_samples = 0;

    for (uint64_t i = 0; i < num_batches; i++) {
        Tensor *batch_x = get_batch(test_images, i, batch_size);
        Tensor *batch_y = get_batch(test_labels, i, batch_size);

        if (batch_x == NULL || batch_y == NULL) {
            if (batch_x != NULL)
                tensor_free(batch_x);
            if (batch_y != NULL)
                tensor_free(batch_y);
            break;
        }

        Tensor *logits = forward_cnn(model, batch_x, false);

        float32_t acc = accuracy(logits, batch_y);
        total_acc += acc * (float32_t)batch_x->shape[0];
        total_samples += batch_x->shape[0];

        tqdm(i + 1, num_batches, "evaluating");
        tensor_free(batch_x);
        tensor_free(batch_y);
        tensor_free(logits);
    }

    return total_acc / (float32_t)total_samples;
}

int32_t main(void) {
    srand(42);
    Tensor *train_images = cifar10_get_train_images();
    Tensor *train_labels = cifar10_get_train_labels();
    Tensor *test_images = cifar10_get_test_images();
    Tensor *test_labels = cifar10_get_test_labels();
    printf("loaded data\n");
    printf("train samples: %" PRIu64 "\n", train_images->shape[0]);
    printf("test samples: %" PRIu64 "\n\n", test_images->shape[0]);

    Model *model = create_model();
    printf("created model\n");
    Tensor **params = NULL;
    size_t param_count = 0;
    get_all_parameters(model, &params, &param_count);
    printf("total params: %zu\n\n", param_count);

    Optimizer *opt = optimizer_adam_create(params, param_count, LEARNING_RATE, 0.9f, 0.999f, 1e-8f, WEIGHT_DECAY);

    printf("starting training\n");
    clock_t s = clock();
    for (uint64_t epoch = 1; epoch <= NUM_EPOCHS; epoch++) {
        train_epoch(model, opt, train_images, train_labels, BATCH_SIZE, epoch);

        float32_t test_acc = evaluate(model, test_images, test_labels, BATCH_SIZE * 2);
        printf("test acc: %.2f%%\n\n", test_acc * 100.0f);
    }

    clock_t end_time = clock();
    double elapsed = (double)(end_time - s) / CLOCKS_PER_SEC;
    printf("training done in %.2f seconds\n", elapsed);

    printf("final cccuracy: %.2f%%\n", evaluate(model, test_images, test_labels, BATCH_SIZE * 2) * 100.0f);

    free(params);
    optimizer_free(opt);
    free_model(model);
    tensor_free(train_images);
    tensor_free(train_labels);
    tensor_free(test_images);
    tensor_free(test_labels);

    return EXIT_SUCCESS;
}
