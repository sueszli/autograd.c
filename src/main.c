#include "cifar10.h"
#include "tqdm.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

extern sample_t test_samples[NUM_TEST_SAMPLES];
extern sample_t train_samples[NUM_TRAIN_SAMPLES];

int32_t main(void) {
    for (int8_t i = 0; i < 5; i++) {
        printf("test sample %d: %s\n", i, label_to_str(test_samples[i].label));
    }

    for (uint64_t i = 0; i < 30; i++) {
        tqdm((i + 1), 30);
        usleep(10000);
    }

    return EXIT_SUCCESS;
}
