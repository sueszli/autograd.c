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

static const char *BATCH_PATHS[6] = {
    "/workspace/data/data_batch_1.bin",
    "/workspace/data/data_batch_2.bin",
    "/workspace/data/data_batch_3.bin",
    "/workspace/data/data_batch_4.bin",
    "/workspace/data/data_batch_5.bin",
    "/workspace/data/test_batch.bin"
};

typedef struct {
    cifar10_sample_t *samples;
    u64 count;
} sample_batch_t;

static u64 get_file_size(FILE *file) {
    assert(fseek(file, 0, SEEK_END) == 0);
    i64 file_size = ftell(file);
    assert(file_size >= 0);
    assert(fseek(file, 0, SEEK_SET) == 0);
    return (u64)file_size;
}

static u64 get_num_samples(u64 file_size) {
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

static sample_batch_t load_samples_from_file(const char *filename) {
    FILE *file = fopen(filename, "rb");
    assert(file != NULL);
    
    defer({ fclose(file); });

    u64 file_size = get_file_size(file);
    u64 num_samples = get_num_samples(file_size);

    cifar10_sample_t *samples = malloc(num_samples * sizeof(cifar10_sample_t));
    assert(samples != NULL);

    for (u64 i = 0; i < num_samples; i++) {
        samples[i] = read_single_sample(file);
    }

    return (sample_batch_t){.samples = samples, .count = num_samples};
}


static sample_batch_t load_train_samples(void) {
    sample_batch_t train_batches[5];
    for (u32 i = 0; i < 5; i++) {
        train_batches[i] = load_samples_from_file(BATCH_PATHS[i]);
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
    
    cifar10_sample_t *merged = malloc(total_samples * sizeof(cifar10_sample_t));
    assert(merged != NULL);

    u64 offset = 0;
    for (u32 i = 0; i < 5; i++) {
        memcpy(merged + offset, train_batches[i].samples, train_batches[i].count * sizeof(cifar10_sample_t));
        offset += train_batches[i].count;
    }
    
    return (sample_batch_t){.samples = merged, .count = total_samples};
}

static void download(void) {
    bool first_file_exist = stat(BATCH_PATHS[0], &(struct stat){0}) == 0;
    if (first_file_exist) {
        return;
    }

    assert(system("mkdir -p /workspace/data") == 0);
    assert(system("curl -L -o /workspace/data/cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz") == 0);
    assert(system("tar -xzf /workspace/data/cifar-10-binary.tar.gz -C /workspace/data --strip-components=1") == 0);
}

cifar10_dataset_t *get_cifar10_dataset(void) {
    download();

    sample_batch_t train_data = load_train_samples();
    sample_batch_t test_data = load_samples_from_file(BATCH_PATHS[5]);
    
    cifar10_dataset_t *dataset = malloc(sizeof(cifar10_dataset_t));
    assert(dataset != NULL);
    
    dataset->train_samples = train_data.samples;
    dataset->num_train_samples = train_data.count;
    dataset->test_samples = test_data.samples;
    dataset->num_test_samples = test_data.count;

    return dataset;
}
