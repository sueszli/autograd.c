#include "cifar10.h"
#include "../utils/defer.h"
#include "../utils/go.h"
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define CIFAR10_RECORD_SIZE (1 + CIFAR10_PIXELS_PER_IMAGE)

static const char *CIFAR10_DATA_DIR = "/workspace/data";
static const char *CIFAR10_DOWNLOAD_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
static const char *CIFAR10_ARCHIVE_PATH = "/workspace/data/cifar-10-binary.tar.gz";
static const char *CIFAR10_BATCH_1_PATH = "/workspace/data/data_batch_1.bin";
static const char *CIFAR10_BATCH_2_PATH = "/workspace/data/data_batch_2.bin";
static const char *CIFAR10_BATCH_3_PATH = "/workspace/data/data_batch_3.bin";
static const char *CIFAR10_BATCH_4_PATH = "/workspace/data/data_batch_4.bin";
static const char *CIFAR10_BATCH_5_PATH = "/workspace/data/data_batch_5.bin";
static const char *CIFAR10_TEST_BATCH_PATH = "/workspace/data/test_batch.bin";

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

    *num_samples = (u64)file_size / CIFAR10_RECORD_SIZE;
    if (file_size % CIFAR10_RECORD_SIZE != 0) {
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

cifar10_dataset_t *get_dataset(void) {
    struct stat st = {0};
    if (stat(CIFAR10_DATA_DIR, &st) == -1) {
        char mkdir_cmd[256];
        snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", CIFAR10_DATA_DIR);
        if (system(mkdir_cmd) != 0) {
            return NULL;
        }
    }

    if (stat(CIFAR10_BATCH_1_PATH, &st) == -1) {
        printf("Downloading CIFAR-10 dataset...\n");
        char wget_cmd[512];
        snprintf(wget_cmd, sizeof(wget_cmd), "wget -P %s %s", CIFAR10_DATA_DIR, CIFAR10_DOWNLOAD_URL);
        if (system(wget_cmd) != 0) {
            return NULL;
        }

        printf("Extracting CIFAR-10 dataset...\n");
        char extract_cmd[512];
        snprintf(extract_cmd, sizeof(extract_cmd), "tar -xzf %s -C %s && mv %s/cifar-10-batches-bin/* %s/ && rmdir %s/cifar-10-batches-bin", CIFAR10_ARCHIVE_PATH, CIFAR10_DATA_DIR, CIFAR10_DATA_DIR, CIFAR10_DATA_DIR, CIFAR10_DATA_DIR);
        if (system(extract_cmd) != 0) {
            return NULL;
        }
    } else {
        printf("CIFAR-10 dataset already exists, skipping download...\n");
    }

    cifar10_dataset_t *dataset = malloc(sizeof(cifar10_dataset_t));
    if (!dataset) {
        return NULL;
    }

    dataset->train_samples = NULL;
    dataset->test_samples = NULL;
    dataset->num_train_samples = 0;
    dataset->num_test_samples = 0;

    printf("Loading training batches into memory...\n");
    const char *batch_paths[] = {CIFAR10_BATCH_1_PATH, CIFAR10_BATCH_2_PATH, CIFAR10_BATCH_3_PATH, CIFAR10_BATCH_4_PATH, CIFAR10_BATCH_5_PATH};

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

    printf("Loading test batch into memory...\n");
    dataset->test_samples = load_batch_samples(CIFAR10_TEST_BATCH_PATH, &dataset->num_test_samples);
    if (!dataset->test_samples) {
        free(dataset->train_samples);
        free(dataset);
        return NULL;
    }

    printf("CIFAR-10 dataset loaded: %zu training samples, %zu test samples\n", dataset->num_train_samples, dataset->num_test_samples);
    return dataset;
}
