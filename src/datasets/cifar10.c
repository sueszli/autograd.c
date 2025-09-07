#include "cifar10.h"
#include "../utils/defer.h"
#include "../utils/go.h"
#include "../utils/types.h"
#include <assert.h>
#include <glob.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static u64 get_num_samples(FILE *file) {
    assert(fseek(file, 0, SEEK_END) == 0);
    i64 file_size = ftell(file);
    assert(file_size >= 0);
    assert(fseek(file, 0, SEEK_SET) == 0);

    i64 record_size = 1 + CIFAR10_PIXELS_PER_IMAGE;
    assert(file_size % record_size == 0);
    return (u64)(file_size / record_size);
}

static sample_arr_t get_samples(const char *filename) {
    FILE *file = fopen(filename, "rb");
    assert(file != NULL);
    defer({ fclose(file); });

    u64 num_samples = get_num_samples(file);
    sample_t *samples = malloc(num_samples * sizeof(sample_t));
    assert(samples != NULL);

    for (u64 i = 0; i < num_samples; i++) {
        sample_t sample;
        assert(fread(&sample.label, 1, 1, file) == 1);
        assert(fread(sample.data, 1, CIFAR10_PIXELS_PER_IMAGE, file) == CIFAR10_PIXELS_PER_IMAGE);
        samples[i] = sample;
    }

    return (sample_arr_t){.samples = samples, .count = num_samples};
}

static bool is_downloaded(void) {
    glob_t glob_result;
    int glob_status = glob("/workspace/data/*.bin", GLOB_NOSORT, NULL, &glob_result);
    defer({ globfree(&glob_result); });
    return glob_status == 0 && glob_result.gl_pathc > 0;
}

static void download(void) {
    assert(!is_downloaded());
    assert(system("mkdir -p /workspace/data") == 0);
    assert(system("curl -L -o /workspace/data/cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz") == 0);
    assert(system("tar -xzf /workspace/data/cifar-10-binary.tar.gz -C /workspace/data --strip-components=1") == 0);
}

sample_arr_t get_test_samples(void) {
    if (!is_downloaded()) {
        download();
    }

    static const char *TEST_BATCH_PATH = "/workspace/data/test_batch.bin";

    return get_samples(TEST_BATCH_PATH);
}

sample_arr_t get_train_samples(void) {
    if (!is_downloaded()) {
        download();
    }

    static const char *TRAIN_BATCH_PATHS[5] = {"/workspace/data/data_batch_1.bin", "/workspace/data/data_batch_2.bin", "/workspace/data/data_batch_3.bin", "/workspace/data/data_batch_4.bin", "/workspace/data/data_batch_5.bin"};

    // load all
    sample_arr_t train_batches[5];
    for (u32 i = 0; i < 5; i++) {
        train_batches[i] = get_samples(TRAIN_BATCH_PATHS[i]);
    }
    defer({
        for (u32 i = 0; i < 5; i++) {
            if (train_batches[i].samples) {
                free(train_batches[i].samples);
            }
        }
    });
    u64 total_samples = 0;
    for (u32 i = 0; i < 5; i++) {
        total_samples += train_batches[i].count;
    }

    // concat
    sample_t *merged = malloc(total_samples * sizeof(sample_t));
    assert(merged != NULL);
    u64 offset = 0;
    for (u32 i = 0; i < 5; i++) {
        memcpy(merged + offset, train_batches[i].samples, train_batches[i].count * sizeof(sample_t));
        offset += train_batches[i].count;
    }

    return (sample_arr_t){.samples = merged, .count = total_samples};
}

//
// utils
//

static const char *CIFAR10_CLASSES[CIFAR10_NUM_CLASSES] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

const char *get_class_name(u8 class_id) {
    assert(class_id < CIFAR10_NUM_CLASSES);
    return CIFAR10_CLASSES[class_id];
}
