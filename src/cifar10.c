#include "cifar10.h"
#include "utils/defer.h"
#include "utils/types.h"
#include <stdlib.h>
#include <string.h>

const char *CLASS_NAMES[CIFAR10_NUM_CLASSES] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

cifar10_batch_t *cifar10_load_batch(const char *filename) {
    FILE *file = fopen(filename, "rb");
    defer(if (file) { fclose(file); });
    if (!file) {
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    i64 file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    u64 num_samples = (u64)file_size / CIFAR10_RECORD_SIZE;
    if (file_size % CIFAR10_RECORD_SIZE != 0) {
        return NULL;
    }

    cifar10_batch_t *batch = malloc(sizeof(cifar10_batch_t));
    if (!batch) {
        return NULL;
    }

    batch->samples = malloc(num_samples * sizeof(cifar10_sample_t));
    if (!batch->samples) {
        free(batch);
        return NULL;
    }
    defer(if (batch && batch->samples) {
        free(batch->samples);
        free(batch);
    });

    batch->num_samples = num_samples;

    for (u64 i = 0; i < num_samples; i++) {
        if (fread(&batch->samples[i].label, 1, 1, file) != 1) {
            return NULL;
        }

        if (fread(batch->samples[i].data, 1, CIFAR10_PIXELS_PER_IMAGE, file) != CIFAR10_PIXELS_PER_IMAGE) {
            return NULL;
        }
    }

    // prevent cleanup by nullifying the pointers in defer scope
    cifar10_sample_t *samples_success = batch->samples;
    cifar10_batch_t *batch_success = batch;
    batch = NULL;
    batch_success->samples = samples_success;
    return batch_success;
}

void cifar10_free_batch(cifar10_batch_t *batch) {
    if (batch) {
        free(batch->samples);
        free(batch);
    }
}

void cifar10_print_sample_info(const cifar10_sample_t *sample) {
    if (sample && sample->label < CIFAR10_NUM_CLASSES) {
        printf("Label: %d (%s)\n", sample->label, CLASS_NAMES[sample->label]);
    }
}

u8 cifar10_get_pixel(const cifar10_sample_t *sample, i32 x, i32 y, i32 channel) {
    if (!sample || x < 0 || x >= CIFAR10_IMAGE_SIZE || y < 0 || y >= CIFAR10_IMAGE_SIZE || channel < 0 || channel >= CIFAR10_NUM_CHANNELS) {
        return 0;
    }

    i32 index = (channel * CIFAR10_IMAGE_SIZE * CIFAR10_IMAGE_SIZE) + (y * CIFAR10_IMAGE_SIZE) + x;
    return sample->data[index];
}