#pragma once

#include "tensor.h"
#include <assert.h>
#include <stdint.h>
#include <string.h>

static inline float32_t accuracy(const Tensor *logits, const Tensor *labels) {
    assert(logits != NULL);
    assert(labels != NULL);
    assert(logits->ndim == 2);
    assert(labels->ndim == 1);

    uint64_t batch_size = logits->shape[0];
    uint64_t num_classes = logits->shape[1];

    uint64_t correct = 0;
    for (uint64_t i = 0; i < batch_size; i++) {
        // argmax
        uint64_t predicted = 0;
        float32_t max_val = logits->data[i * num_classes];
        for (uint64_t j = 1; j < num_classes; j++) {
            float32_t val = logits->data[i * num_classes + j];
            if (val > max_val) {
                max_val = val;
                predicted = j;
            }
        }

        uint64_t true_label = (uint64_t)labels->data[i];
        if (predicted == true_label) {
            correct++;
        }
    }

    return (float32_t)correct / (float32_t)batch_size;
}
