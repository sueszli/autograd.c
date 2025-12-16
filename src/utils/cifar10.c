#include "cifar10.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//
// data loader from disk
//

// path to subdirectory containing the downloaded data files
#define DATA_DIRECTORY "data"

static uint8_t train_images[NUM_TRAIN_SAMPLES * INPUT_SIZE];
static label_t train_labels[NUM_TRAIN_SAMPLES];
static uint8_t test_images[NUM_TEST_SAMPLES * INPUT_SIZE];
static label_t test_labels[NUM_TEST_SAMPLES];

static void load_batch(const char *filepath, uint8_t *images_out, label_t *labels_out, int32_t count) {
    assert(filepath != NULL);
    assert(images_out != NULL);
    assert(labels_out != NULL);
    assert(count > 0);

    FILE *f = fopen(filepath, "rb");
    assert(f != NULL && "failed to open batch file");

    for (int32_t i = 0; i < count; i++) {
        uint8_t label;
        int64_t label_read = (int64_t)fread(&label, 1, 1, f);
        assert(label_read == 1 && "failed to read label");
        assert(label < NUM_CLASSES && "invalid label");

        labels_out[i] = (label_t)label;

        int64_t read = (int64_t)fread(images_out + (i * INPUT_SIZE), 1, INPUT_SIZE, f);
        assert(read == INPUT_SIZE && "failed to read image data");
    }

    int32_t close_result = fclose(f);
    assert(close_result == 0 && "failed to close batch file");
}

// constructor to load data on program start
__attribute__((constructor)) static void load_data(void) {
    static const char *const batches[] = {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin"};
    assert(NUM_TRAIN_SAMPLES == 50000);
    assert(NUM_TEST_SAMPLES == 10000);

    int32_t samples_per_batch = NUM_TRAIN_SAMPLES / 5;
    assert(samples_per_batch == 10000);

    // load train data
    for (int32_t i = 0; i < 5; i++) {
        char path[512];
        int32_t written = snprintf(path, sizeof(path), "%s/%s", DATA_DIRECTORY, batches[i]);
        assert(written > 0 && written < (int32_t)sizeof(path) && "path buffer overflow");

        int32_t offset = i * samples_per_batch;
        load_batch(path, train_images + (offset * INPUT_SIZE), train_labels + offset, samples_per_batch);
    }

    // load test data
    char path[512];
    int32_t written = snprintf(path, sizeof(path), "%s/test_batch.bin", DATA_DIRECTORY);
    assert(written > 0 && written < (int32_t)sizeof(path) && "path buffer overflow");
    load_batch(path, test_images, test_labels, NUM_TEST_SAMPLES);
}

static Tensor *images_to_tensor(const uint8_t *data, uint64_t count) {
    uint64_t total_elements = count * INPUT_SIZE;
    const uint64_t shape[] = {count, CHANNELS, HEIGHT, WIDTH};
    Tensor *t = tensor_zeros(shape, 4, false);
    assert(t != NULL);

    for (uint64_t i = 0; i < total_elements; i++) {
        t->data[i] = (float32_t)data[i] / 255.0f; // normalize in place
    }
    return t;
}

static Tensor *labels_to_tensor(const label_t *labels, uint64_t count) {
    const uint64_t shape[] = {count};
    Tensor *t = tensor_zeros(shape, 1, false);
    assert(t != NULL);

    for (uint64_t i = 0; i < count; i++) {
        t->data[i] = (float32_t)labels[i];
    }
    return t;
}

Tensor *cifar10_get_train_images(void) { return images_to_tensor(train_images, NUM_TRAIN_SAMPLES); }

Tensor *cifar10_get_train_labels(void) { return labels_to_tensor(train_labels, NUM_TRAIN_SAMPLES); }

Tensor *cifar10_get_test_images(void) { return images_to_tensor(test_images, NUM_TEST_SAMPLES); }

Tensor *cifar10_get_test_labels(void) { return labels_to_tensor(test_labels, NUM_TEST_SAMPLES); }

//
// utils
//

const char *label_to_str(label_t label) {
    static const char *const labels[] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
    assert(label >= 0 && label < NUM_CLASSES);
    return labels[label];
}

Tensor *get_batch(const Tensor *data, uint64_t batch_idx, uint64_t batch_size) {
    assert(data != NULL);
    assert(data->ndim >= 1);

    uint64_t total_samples = data->shape[0];
    uint64_t start = batch_idx * batch_size;

    if (start >= total_samples) {
        return NULL;
    }

    uint64_t actual_batch = (start + batch_size > total_samples) ? (total_samples - start) : batch_size; // edge case for last batch
    uint64_t elements_per_sample = data->size / total_samples;
    uint64_t batch_elements = actual_batch * elements_per_sample;

    float32_t *batch_data = (float32_t *)malloc(batch_elements * sizeof(float32_t));
    assert(batch_data != NULL && "malloc failed");
    memcpy(batch_data, &data->data[start * elements_per_sample], batch_elements * sizeof(float32_t));

    uint64_t *batch_shape = (uint64_t *)malloc(data->ndim * sizeof(uint64_t));
    assert(batch_shape != NULL && "malloc failed");
    memcpy(batch_shape, data->shape, data->ndim * sizeof(uint64_t));
    batch_shape[0] = actual_batch;

    Tensor *batch = tensor_create(batch_data, batch_shape, data->ndim, false);
    free(batch_data);
    free(batch_shape);

    return batch;
}
