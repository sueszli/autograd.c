#pragma once

#include "../utils/types.h"

#define CIFAR10_IMAGE_SIZE 32
#define CIFAR10_NUM_CHANNELS 3
#define CIFAR10_PIXELS_PER_IMAGE (CIFAR10_IMAGE_SIZE * CIFAR10_IMAGE_SIZE * CIFAR10_NUM_CHANNELS)
#define CIFAR10_NUM_CLASSES 10

typedef struct {
    u8 label;
    u8 data[CIFAR10_PIXELS_PER_IMAGE];
} cifar10_sample_t;

typedef struct {
    cifar10_sample_t *train_samples;
    cifar10_sample_t *test_samples;
    u64 num_train_samples;
    u64 num_test_samples;
} cifar10_dataset_t;

static const char *CIFAR10_CLASS_NAMES[CIFAR10_NUM_CLASSES] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

cifar10_dataset_t *get_dataset(void);