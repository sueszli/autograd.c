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
    samples_count_t train_samples = get_train_samples();
    samples_count_t test_samples = get_test_samples();
    assert(train_samples.samples != NULL);
    assert(test_samples.samples != NULL);

    printf("loaded %lu training samples and %lu test samples.\n", train_samples.count, test_samples.count);

    sample_t first_sample = train_samples.samples[0];
    printf("first sample label: %u (%s)\n", first_sample.label, get_class_name(first_sample.label));

    return EXIT_SUCCESS;
}
