#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "cifar10.h"

int main(void) {
    printf("Loading CIFAR-10 data...\n");

    cifar10_batch_t *batch = cifar10_load_batch("/workspace/data/data_batch_1.bin");
    if (!batch) {
        printf("Failed to load CIFAR-10 batch\n");
        return EXIT_FAILURE;
    }

    printf("Loaded %zu samples\n", batch->num_samples);

    for (int i = 0; i < 5 && i < (int)batch->num_samples; i++) {
        printf("\nSample %d:\n", i);
        cifar10_print_sample_info(&batch->samples[i]);

        printf("First few RGB values: ");
        for (int j = 0; j < 9; j++) {
            printf("%d ", batch->samples[i].data[j]);
        }
        printf("...\n");
    }

    cifar10_free_batch(batch);
    return EXIT_SUCCESS;
}
