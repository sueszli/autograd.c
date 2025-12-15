#pragma once

#include "tensor.h"
#include <stdint.h>

#define CHANNELS 3
#define HEIGHT 32
#define WIDTH 32
#define INPUT_SIZE (CHANNELS * HEIGHT * WIDTH)
#define NUM_TRAIN_SAMPLES 50000
#define NUM_TEST_SAMPLES 10000

#define NUM_CLASSES 10
typedef enum { AIRPLANE = 0, AUTOMOBILE = 1, BIRD = 2, CAT = 3, DEER = 4, DOG = 5, FROG = 6, HORSE = 7, SHIP = 8, TRUCK = 9 } label_t;

Tensor *cifar10_get_train_images(void);
Tensor *cifar10_get_train_labels(void);
Tensor *cifar10_get_test_images(void);
Tensor *cifar10_get_test_labels(void);

const char *label_to_str(label_t label);
