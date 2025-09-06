#include "cifar10.h"
#include "../utils/defer.h"
#include "../utils/go.h"
#include "../utils/types.h"
#include <assert.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *BATCH_1_PATH = "/workspace/data/data_batch_1.bin";
static const char *BATCH_2_PATH = "/workspace/data/data_batch_2.bin";
static const char *BATCH_3_PATH = "/workspace/data/data_batch_3.bin";
static const char *BATCH_4_PATH = "/workspace/data/data_batch_4.bin";
static const char *BATCH_5_PATH = "/workspace/data/data_batch_5.bin";
static const char *BATCH_TEST_PATH = "/workspace/data/test_batch.bin";

static void download(void) {
    bool exists = stat(BATCH_1_PATH, &(struct stat){0}) == 0;
    if (exists) {
        return;
    }

    const char *mkdir_cmd = "mkdir -p /workspace/data";
    const char *curl_cmd = "curl -L -o /workspace/data/cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
    const char *tar_cmd = "tar -xzf /workspace/data/cifar-10-binary.tar.gz -C /workspace/data --strip-components=1";
    assert(system(mkdir_cmd) == 0);
    assert(system(curl_cmd) == 0);
    assert(system(tar_cmd) == 0);
}

typedef struct {
    cifar10_sample_t *samples;
    u64 count;
} batch_data_t;

typedef struct {
    const char *path;
    batch_data_t data;
} batch_loader_t;

static u64 get_file_size(FILE *file) {
    assert(fseek(file, 0, SEEK_END) == 0);
    i64 file_size = ftell(file);
    assert(file_size >= 0);
    assert(fseek(file, 0, SEEK_SET) == 0);
    return (u64)file_size;
}

static u64 calculate_num_samples(u64 file_size) {
    u64 record_size = 1 + CIFAR10_PIXELS_PER_IMAGE;
    assert(file_size % record_size == 0);
    return file_size / record_size;
}

static cifar10_sample_t read_single_sample(FILE *file) {
    cifar10_sample_t sample;
    assert(fread(&sample.label, 1, 1, file) == 1);
    assert(fread(sample.data, 1, CIFAR10_PIXELS_PER_IMAGE, file) == CIFAR10_PIXELS_PER_IMAGE);
    return sample;
}

static batch_data_t load_batch_samples(const char *filename) {
    FILE *file = fopen(filename, "rb");
    assert(file != NULL);
    
    defer({ fclose(file); });

    u64 file_size = get_file_size(file);
    u64 num_samples = calculate_num_samples(file_size);

    cifar10_sample_t *samples = malloc(num_samples * sizeof(cifar10_sample_t));
    assert(samples != NULL);

    for (u64 i = 0; i < num_samples; i++) {
        samples[i] = read_single_sample(file);
    }

    return (batch_data_t){.samples = samples, .count = num_samples};
}

static batch_data_t load_batch_worker(const char *path) {
    return load_batch_samples(path);
}

static batch_loader_t create_batch_loader(const char *path) {
    return (batch_loader_t){.path = path, .data = {.samples = NULL, .count = 0}};
}

static batch_loader_t load_batch(batch_loader_t loader) {
    loader.data = load_batch_worker(loader.path);
    return loader;
}

static u64 calculate_total_samples(const batch_loader_t loaders[5]) {
    u64 total = 0;
    for (u32 i = 0; i < 5; i++) {
        total += loaders[i].data.count;
    }
    return total;
}

static cifar10_sample_t *merge_batches(const batch_loader_t loaders[5], u64 total_samples) {
    cifar10_sample_t *merged = malloc(total_samples * sizeof(cifar10_sample_t));
    assert(merged != NULL);

    u64 offset = 0;
    for (u32 i = 0; i < 5; i++) {
        memcpy(merged + offset, loaders[i].data.samples, loaders[i].data.count * sizeof(cifar10_sample_t));
        offset += loaders[i].data.count;
    }
    return merged;
}

static void cleanup_loaders(const batch_loader_t loaders[5]) {
    for (u32 i = 0; i < 5; i++) {
        if (loaders[i].data.samples) {
            free(loaders[i].data.samples);
        }
    }
}

cifar10_dataset_t *get_cifar10_dataset(void) {
    download();

    const char *batch_paths[] = {BATCH_1_PATH, BATCH_2_PATH, BATCH_3_PATH, BATCH_4_PATH, BATCH_5_PATH};
    batch_loader_t loaders[5];
    
    for (u32 i = 0; i < 5; i++) {
        loaders[i] = create_batch_loader(batch_paths[i]);
        loaders[i] = load_batch(loaders[i]);
    }
    
    defer({ cleanup_loaders(loaders); });
    
    u64 total_samples = calculate_total_samples(loaders);
    cifar10_sample_t *train_samples = merge_batches(loaders, total_samples);
    
    batch_data_t test_data = load_batch_samples(BATCH_TEST_PATH);
    
    cifar10_dataset_t *dataset = malloc(sizeof(cifar10_dataset_t));
    assert(dataset != NULL);
    
    dataset->train_samples = train_samples;
    dataset->num_train_samples = total_samples;
    dataset->test_samples = test_data.samples;
    dataset->num_test_samples = test_data.count;

    return dataset;
}
