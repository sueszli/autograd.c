#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "datasets/cifar10.h"
#include "utils/types.h"

int main(void) {
    cifar10_dataset_t *dataset = get_dataset();
    if (!dataset) {
        printf("Failed to get CIFAR-10 dataset\n");
        return EXIT_FAILURE;
    }

    printf("CIFAR-10 loaded: %zu train, %zu test\n", dataset->num_train_samples, dataset->num_test_samples);
    printf("First sample: %s\n", CIFAR10_CLASS_NAMES[dataset->train_samples[0].label]);

    free(dataset->train_samples);
    free(dataset->test_samples);
    free(dataset);
    return EXIT_SUCCESS;
}