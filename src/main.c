#include "cifar10.h"
#include "tensor.h"
#include "tqdm.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int32_t main(void) {
    float32_t float_data[INPUT_SIZE];
    uint64_t shape[] = {CHANNELS, HEIGHT, WIDTH};

    // load 5 images into tensors
    for (int8_t i = 0; i < 5; i++) {
        sample_t *s = &test_samples[i];
        printf("test sample %d: %s\n", i, label_to_str(s->label));

        for (uint64_t j = 0; j < INPUT_SIZE; j++) {
            float_data[j] = (float32_t)s->data[j] / 255.0f;
        }

        Tensor *t = tensor_create(float_data, shape, 3, false);
        tensor_print(t);
        tensor_free(t);
    }

    // demo tqdm
    for (uint64_t i = 0; i < 30; i++) {
        tqdm((i + 1), 30);
        usleep(10000);
    }

    return EXIT_SUCCESS;
}
