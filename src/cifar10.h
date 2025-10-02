#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_CLASSES 10
#define INPUT_SIZE (32 * 32 * 3)
#define NUM_TRAIN_SAMPLES 50000
#define NUM_TEST_SAMPLES 10000

typedef enum { AIRPLANE = 0, AUTOMOBILE = 1, BIRD = 2, CAT = 3, DEER = 4, DOG = 5, FROG = 6, HORSE = 7, SHIP = 8, TRUCK = 9 } label_t;

typedef struct {
    label_t label;
    uint8_t data[INPUT_SIZE];
} sample_t;

typedef sample_t train_samples_t[NUM_TRAIN_SAMPLES];
typedef sample_t test_samples_t[NUM_TEST_SAMPLES];

static inline const char *label_to_str(label_t label) {
    static const char *labels[] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
    assert(label < NUM_CLASSES && "invalid label");
    return labels[label];
}

static inline void load_batch(const char *filepath, sample_t *samples, int32_t count) {
    FILE *f = fopen(filepath, "rb");
    assert(f && "failed to open batch file");

    for (int32_t i = 0; i < count; i++) {
        uint8_t label;
        size_t label_read = fread(&label, 1, 1, f);
        assert(label_read == 1 && "failed to read label");
        assert(label < NUM_CLASSES && "invalid label");
        samples[i].label = (label_t)label;
        size_t data_read = fread(samples[i].data, 1, INPUT_SIZE, f);
        assert(data_read == INPUT_SIZE && "failed to read image data");
    }

    fclose(f);
}

static inline void load_train_samples(const char *data_dir, train_samples_t samples) {
    assert(data_dir && samples);
    const char *batches[] = {"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin"};

    for (int8_t i = 0; i < 5; i++) {
        char path[512];
        int32_t written = snprintf(path, sizeof(path), "%s/%s", data_dir, batches[i]);
        assert(written > 0 && written < (int16_t)sizeof(path));
        load_batch(path, samples + i * 10000, 10000);
    }
}

static inline void load_test_samples(const char *data_dir, test_samples_t samples) {
    assert(data_dir && samples);
    char path[512];
    int32_t written = snprintf(path, sizeof(path), "%s/test_batch.bin", data_dir);
    assert(written > 0 && written < (int16_t)sizeof(path));
    load_batch(path, samples, NUM_TEST_SAMPLES);
}
