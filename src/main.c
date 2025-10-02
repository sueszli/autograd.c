#include "cifar10.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int32_t main(void) {
    const char *data_path = get_data_path();

    test_samples_t *test_samples = malloc(sizeof(test_samples_t));
    assert(test_samples);
    load_test_samples(data_path, *test_samples);

    printf("Test samples:\n");
    for (int8_t i = 0; i < 10; i++) {
        printf("  Sample %d: %s\n", i, label_to_str((*test_samples)[i].label));
    }
    free(test_samples);

    train_samples_t *train_samples = malloc(sizeof(train_samples_t));
    assert(train_samples);
    load_train_samples(data_path, *train_samples);

    printf("\nTrain samples:\n");
    for (int8_t i = 0; i < 10; i++) {
        printf("  Sample %d: %s\n", i, label_to_str((*train_samples)[i].label));
    }
    free(train_samples);

    return EXIT_SUCCESS;
}
