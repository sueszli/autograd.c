#include "cifar10.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int32_t main(void) {
    char cwd[512];
    char *cwd_result = getcwd(cwd, sizeof(cwd));
    assert(cwd_result);

    char data_path[512];
    int32_t written = snprintf(data_path, sizeof(data_path), "%s/data", cwd);
    assert(written > 0 && written < (int16_t)sizeof(data_path));

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
