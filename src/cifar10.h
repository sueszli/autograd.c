#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_CLASSES 10
typedef enum { AIRPLANE = 0, AUTOMOBILE = 1, BIRD = 2, CAT = 3, DEER = 4, DOG = 5, FROG = 6, HORSE = 7, SHIP = 8, TRUCK = 9 } label_t;

#define CHANNELS 3
#define HEIGHT 32
#define WIDTH 32
#define INPUT_SIZE (CHANNELS * HEIGHT * WIDTH)
#define NUM_TRAIN_SAMPLES 50000
#define NUM_TEST_SAMPLES 10000

//
// data loading
//

// path to subdirectory with dataset
#define DATA_DIRECTORY "data"

// static arrays with dataset
typedef struct {
    label_t label;
    uint8_t data[INPUT_SIZE];
} sample_t;
static sample_t test_samples[NUM_TEST_SAMPLES];
static sample_t train_samples[NUM_TRAIN_SAMPLES];

static inline void load_batch(const char *filepath, sample_t *samples, int32_t count) {
    assert(filepath != NULL);
    assert(samples != NULL);
    assert(count > 0);
    assert(count <= NUM_TRAIN_SAMPLES);

    FILE *f = fopen(filepath, "rb");
    assert(f != NULL);

    for (int32_t i = 0; i < count; i++) {
        uint8_t label;
        int64_t label_read = (int64_t)fread(&label, 1, 1, f);
        assert(label_read == 1 && "failed to read label");
        assert(label < NUM_CLASSES && "invalid label");
        samples[i].label = (label_t)label;

        int64_t read = (int64_t)fread(samples[i].data, 1, INPUT_SIZE, f);
        assert(read == INPUT_SIZE && "failed to read image data");
        assert(samples[i].label < NUM_CLASSES && "label corruption detected");
    }

    int32_t close_result = fclose(f);
    assert(close_result == 0 && "failed to close batch file");
}

// load data from disk into static arrays
__attribute__((constructor)) // compiler trick: called on header import
static void load_data(void) {
    static const char *const batches[] = {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin"};
    assert(NUM_TRAIN_SAMPLES == 50000);
    assert(NUM_TEST_SAMPLES == 10000);

    int32_t samples_per_batch = NUM_TRAIN_SAMPLES / 5;
    assert(samples_per_batch == 10000);
    assert(samples_per_batch * 5 == NUM_TRAIN_SAMPLES);

    int32_t const max_batches = 5;
    for (int32_t i = 0; i < max_batches; i++) {
        assert(batches[i] != NULL);

        char path[512];
        int32_t written = snprintf(path, sizeof(path), "%s/%s", DATA_DIRECTORY, batches[i]);
        assert(written > 0 && written < (int32_t)sizeof(path) && "path buffer overflow");

        int32_t offset = i * samples_per_batch;
        assert(offset >= 0 && offset < NUM_TRAIN_SAMPLES && "sample offset out of bounds");
        assert(offset + samples_per_batch <= NUM_TRAIN_SAMPLES);

        load_batch(path, train_samples + offset, samples_per_batch);
    }

    char path[512];
    int32_t written = snprintf(path, sizeof(path), "%s/test_batch.bin", DATA_DIRECTORY);
    assert(written > 0 && written < (int32_t)sizeof(path) && "path buffer overflow");
    load_batch(path, test_samples, NUM_TEST_SAMPLES);
}

//
// utils
//

static inline const char *label_to_str(label_t label) {
    static const char *const labels[] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

    assert(label >= 0 && "invalid label");
    assert(label < NUM_CLASSES && "invalid label");
    assert((int32_t)(sizeof(labels) / sizeof(labels[0])) == NUM_CLASSES && "labels array size mismatch");

    const char *result = labels[label];
    assert(result != NULL);

    return result;
}
