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
    const char *path;
    cifar10_sample_t *samples;
    u64 num_samples;
    _Atomic bool loaded;
} batch_loader_t;

static cifar10_sample_t *load_batch_samples(const char *filename, u64 *num_samples) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    i64 file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    u64 record_size = 1 + CIFAR10_PIXELS_PER_IMAGE;
    *num_samples = (u64)file_size / record_size;
    if ((u64)file_size % record_size != 0) {
        fclose(file);
        return NULL;
    }

    cifar10_sample_t *samples = malloc(*num_samples * sizeof(cifar10_sample_t));
    if (!samples) {
        fclose(file);
        return NULL;
    }

    for (u64 i = 0; i < *num_samples; i++) {
        if (fread(&samples[i].label, 1, 1, file) != 1) {
            free(samples);
            fclose(file);
            return NULL;
        }

        if (fread(samples[i].data, 1, CIFAR10_PIXELS_PER_IMAGE, file) != CIFAR10_PIXELS_PER_IMAGE) {
            free(samples);
            fclose(file);
            return NULL;
        }
    }

    fclose(file);
    return samples;
}

static void load_batch_worker(batch_loader_t *loader) {
    loader->samples = load_batch_samples(loader->path, &loader->num_samples);
    atomic_store(&loader->loaded, loader->samples != NULL);
}

cifar10_dataset_t *get_cifar10_dataset(void) {
    download();

    cifar10_dataset_t *dataset = malloc(sizeof(cifar10_dataset_t));
    assert(dataset != NULL);

    dataset->train_samples = NULL;
    dataset->test_samples = NULL;
    dataset->num_train_samples = 0;
    dataset->num_test_samples = 0;
    const char *batch_paths[] = {BATCH_1_PATH, BATCH_2_PATH, BATCH_3_PATH, BATCH_4_PATH, BATCH_5_PATH};

    batch_loader_t loaders[5];
    defer({
        for (u32 i = 0; i < 5; i++) {
            if (atomic_load(&loaders[i].loaded) && loaders[i].samples) {
                free(loaders[i].samples);
            }
        }
    });

    for (u32 i = 0; i < 5; i++) {
        loaders[i].path = batch_paths[i];
        loaders[i].samples = NULL;
        loaders[i].num_samples = 0;
        atomic_store(&loaders[i].loaded, false);

        load_batch_worker(&loaders[i]);
    }

    u64 total_samples = 0;
    for (u32 i = 0; i < 5; i++) {
        if (!atomic_load(&loaders[i].loaded) || !loaders[i].samples) {
            free(dataset);
            return NULL;
        }
        total_samples += loaders[i].num_samples;
    }

    dataset->train_samples = malloc(total_samples * sizeof(cifar10_sample_t));
    if (!dataset->train_samples) {
        free(dataset);
        return NULL;
    }

    u64 offset = 0;
    for (u32 i = 0; i < 5; i++) {
        memcpy(dataset->train_samples + offset, loaders[i].samples, loaders[i].num_samples * sizeof(cifar10_sample_t));
        offset += loaders[i].num_samples;
    }
    dataset->num_train_samples = total_samples;

    dataset->test_samples = load_batch_samples(BATCH_TEST_PATH, &dataset->num_test_samples);
    if (!dataset->test_samples) {
        free(dataset->train_samples);
        free(dataset);
        return NULL;
    }

    return dataset;
}
