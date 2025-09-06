#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "datasets/cifar10.h"
#include "utils/defer.h"
#include "utils/types.h"

int main(void) {
    cifar10_dataset_t *dataset = get_cifar10_dataset();
    assert(dataset != NULL);
    defer({
        free(dataset->train_samples);
        free(dataset->test_samples);
        free(dataset);
    });

    printf("loaded: %zu train, %zu test\n", dataset->num_train_samples, dataset->num_test_samples);
    printf("first sample: %s\n", CIFAR10_CLASS_NAMES[dataset->train_samples[0].label]);

    return EXIT_SUCCESS;
}
