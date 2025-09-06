#pragma once

#include "utils/types.h"
#include <stdint.h>
#include <stdio.h>

#define CIFAR10_IMAGE_SIZE 32
#define CIFAR10_NUM_CHANNELS 3
#define CIFAR10_PIXELS_PER_IMAGE (CIFAR10_IMAGE_SIZE * CIFAR10_IMAGE_SIZE * CIFAR10_NUM_CHANNELS)
#define CIFAR10_RECORD_SIZE (1 + CIFAR10_PIXELS_PER_IMAGE)
#define CIFAR10_NUM_CLASSES 10
#define CIFAR10_SAMPLES_PER_BATCH 10000

extern const char *CLASS_NAMES[CIFAR10_NUM_CLASSES];

typedef struct {
    u8 label;
    u8 data[CIFAR10_PIXELS_PER_IMAGE];
} cifar10_sample_t;

typedef struct {
    cifar10_sample_t *samples;
    u64 num_samples;
} cifar10_batch_t;

cifar10_batch_t *cifar10_load_batch(const char *filename);
void cifar10_free_batch(cifar10_batch_t *batch);
void cifar10_print_sample_info(const cifar10_sample_t *sample);
u8 cifar10_get_pixel(const cifar10_sample_t *sample, i32 x, i32 y, i32 channel);
