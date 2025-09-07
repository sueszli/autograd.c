#pragma once

#include "../utils/types.h"

#define CIFAR10_IMAGE_SIZE 32
#define CIFAR10_NUM_CHANNELS 3
#define CIFAR10_NUM_CLASSES 10
#define CIFAR10_PIXELS_PER_IMAGE (CIFAR10_IMAGE_SIZE * CIFAR10_IMAGE_SIZE * CIFAR10_NUM_CHANNELS)

typedef struct {
    u8 label;
    u8 data[CIFAR10_PIXELS_PER_IMAGE];
} sample_t;

typedef struct {
    sample_t *samples;
    u64 count;
} samples_count_t;

samples_count_t get_test_samples(void);
samples_count_t get_train_samples(void);

char *get_class_name(u8 class_id);
