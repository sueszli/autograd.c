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

    i64 record_size = 1 + INPUT_SIZE;
    assert(file_size % record_size == 0);
    return (u64)(file_size / record_size);
}

static u64 load_samples_to_buffer(const char *filename, sample_t *buffer, u64 max_samples) {
    FILE *file = fopen(filename, "rb");
    assert(file != NULL);
    defer({ fclose(file); });

    u64 num_samples = get_num_samples(file);
    assert(num_samples <= max_samples);

    for (u64 i = 0; i < num_samples; i++) {
        assert(fread(&buffer[i].label, 1, 1, file) == 1);
        assert(fread(buffer[i].data, 1, INPUT_SIZE, file) == INPUT_SIZE);
    }

    return num_samples;
}

static bool is_downloaded(void) {
    glob_t glob_result;
    i32 glob_status = glob("/workspace/data/*.bin", GLOB_NOSORT, NULL, &glob_result);
    defer({ globfree(&glob_result); });
    return glob_status == 0 && glob_result.gl_pathc > 0;
}

static void download(void) {
    assert(!is_downloaded());
    assert(system("mkdir -p /workspace/data") == 0);
    assert(system("curl -L -o /workspace/data/cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz") == 0);
    assert(system("tar -xzf /workspace/data/cifar-10-binary.tar.gz -C /workspace/data --strip-components=1") == 0);
}

void load_test_samples_to_buffer(test_samples_t samples) {
    if (!is_downloaded()) {
        download();
    }

    const char *TEST_BATCH_PATH = "/workspace/data/test_batch.bin";
    load_samples_to_buffer(TEST_BATCH_PATH, samples, NUM_TEST_SAMPLES);
}

void load_train_samples_to_buffer(train_samples_t samples) {
    if (!is_downloaded()) {
        download();
    }

    const char *TRAIN_BATCH_PATHS[5] = {"/workspace/data/data_batch_1.bin", "/workspace/data/data_batch_2.bin", "/workspace/data/data_batch_3.bin", "/workspace/data/data_batch_4.bin", "/workspace/data/data_batch_5.bin"};
    u64 offset = 0;
    for (u32 i = 0; i < 5; i++) {
        u64 batch_samples = load_samples_to_buffer(TRAIN_BATCH_PATHS[i], samples + offset, NUM_TRAIN_SAMPLES - offset);
        offset += batch_samples;
        assert(offset <= NUM_TRAIN_SAMPLES);
    }
}

const char *get_class_name(cifar10_class_t class_id) {
    assert(class_id < NUM_CLASSES);
    static const char *CIFAR10_CLASSES[NUM_CLASSES] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
    return CIFAR10_CLASSES[class_id];
}
