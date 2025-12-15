#include "dataloader/cifar10.h"
#include "tensor.h"
#include "utils/tqdm.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int32_t main(void) {

    const uint64_t shape[] = {CHANNELS, HEIGHT, WIDTH};

    Tensor *images = cifar10_get_test_images();
    Tensor *labels = cifar10_get_test_labels();

    for (uint64_t i = 0; i < 5; i++) {
        label_t idx = (label_t)labels->data[i];
        printf("test sample %" PRIu64 ": %s\n", i, label_to_str(idx));

        const float32_t *img_data = &images->data[i * INPUT_SIZE];
        Tensor *t = tensor_create(img_data, shape, CHANNELS, false);
        tensor_print(t);
        printf("\n");
        tensor_print(t);
        tensor_free(t);
    }

    tensor_free(images);
    tensor_free(labels);

    // demo tqdm
    for (uint64_t i = 0; i < 30; i++) {
        tqdm((i + 1), 30);
        usleep(10000);
    }

    return EXIT_SUCCESS;
}
