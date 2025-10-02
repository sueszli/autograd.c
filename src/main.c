#include "cifar10.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static test_samples_t test_samples;
static train_samples_t train_samples;

int32_t main(void) {
    load_test_samples(test_samples);

    printf("Test samples:\n");
    for (int8_t i = 0; i < 10; i++) {
        printf("\tSample %d: %s\n", i, label_to_str(test_samples[i].label));
    }

    load_train_samples(train_samples);

    printf("\nTrain samples:\n");
    for (int8_t i = 0; i < 10; i++) {
        printf("\tSample %d: %s\n", i, label_to_str(train_samples[i].label));
    }

    return EXIT_SUCCESS;
}
