#pragma once

#include "../utils/types.h"

#define NUM_CLASSES 10
#define INPUT_SIZE (32 * 32 * 3)
#define NUM_TRAIN_SAMPLES 50000
#define NUM_TEST_SAMPLES 10000

typedef enum { AIRPLANE = 0, AUTOMOBILE = 1, BIRD = 2, CAT = 3, DEER = 4, DOG = 5, FROG = 6, HORSE = 7, SHIP = 8, TRUCK = 9 } cifar10_class_t;

typedef struct {
    cifar10_class_t label;
    u8 data[INPUT_SIZE];
} sample_t;

typedef sample_t train_samples_t[NUM_TRAIN_SAMPLES];
typedef sample_t test_samples_t[NUM_TEST_SAMPLES];

void load_test_samples_to_buffer(test_samples_t samples);
void load_train_samples_to_buffer(train_samples_t samples);

const char *get_class_name(cifar10_class_t class_id);
