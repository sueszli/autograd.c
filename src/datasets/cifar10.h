#pragma once

#include "../utils/types.h"

#define NUM_CLASSES 10
#define NUM_PIXELS (32 * 32 * 3)
#define NUM_TRAIN_SAMPLES 50000
#define NUM_TEST_SAMPLES 10000

static const char *CIFAR10_CLASSES[NUM_CLASSES] __attribute__((unused)) = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

typedef struct {
    u8 label;
    u8 data[NUM_PIXELS];
} sample_t;

typedef sample_t train_samples_t[NUM_TRAIN_SAMPLES];
typedef sample_t test_samples_t[NUM_TEST_SAMPLES];

void load_test_samples_to_buffer(test_samples_t samples);
void load_train_samples_to_buffer(train_samples_t samples);

const char *get_class_name(u8 class_id);
