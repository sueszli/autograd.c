#pragma once

#include "../utils/types.h"

#define NUM_CLASSES 10
#define NUM_PIXELS (32 * 32 * 3)
#define NUM_TRAIN_SAMPLES 50000
#define NUM_TEST_SAMPLES 10000

typedef struct {
    u8 label;
    u8 data[NUM_PIXELS];
} sample_t;

typedef sample_t train_samples_t[NUM_TRAIN_SAMPLES];
typedef sample_t test_samples_t[NUM_TEST_SAMPLES];

void get_test_samples(test_samples_t samples);
void get_train_samples(train_samples_t samples);

const char *get_class_name(u8 class_id);
