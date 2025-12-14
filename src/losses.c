#include "losses.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define EPSILON 1e-7f

float32_t mse_loss(const Tensor *predictions, const Tensor *targets) {
    assert(predictions != NULL);
    assert(targets != NULL);
    assert(predictions->data != NULL || predictions->size == 0);
    assert(targets->data != NULL || targets->size == 0);
    assert(predictions->size == targets->size);
    assert(predictions->ndim == targets->ndim);
    for (uint64_t i = 0; i < predictions->ndim; i++) {
        assert(predictions->shape[i] == targets->shape[i]);
    }

    float32_t sum_squared_error = 0.0f;
    for (uint64_t i = 0; i < predictions->size; i++) {
        float32_t diff = predictions->data[i] - targets->data[i];
        sum_squared_error += diff * diff;
    }
    return sum_squared_error / (float32_t)predictions->size;
}

float32_t cross_entropy_loss(const Tensor *logits, const Tensor *targets) {
    assert(logits != NULL);
    assert(targets != NULL);
    assert(logits->data != NULL || logits->size == 0);
    assert(targets->data != NULL || targets->size == 0);
    assert(logits->ndim == 2);
    assert(targets->ndim == 1);
    assert(logits->shape[0] == targets->shape[0]);

    uint64_t batch_size = logits->shape[0];
    uint64_t num_classes = logits->shape[1];

    float32_t sum_loss = 0.0f;

    for (uint64_t i = 0; i < batch_size; i++) {
        // Get target class index for this sample
        float32_t target_float = targets->data[i];
        // Ensure target is a valid integer index
        assert(target_float >= 0.0f && target_float < (float32_t)num_classes);
        // We cast to int safely because we just checked range
        uint64_t target_idx = (uint64_t)target_float;

        // Find max logit for numerical stability (Log-Sum-Exp trick)
        float32_t max_logit = -FLT_MAX;
        for (uint64_t j = 0; j < num_classes; j++) {
            float32_t logit = logits->data[i * num_classes + j];
            if (logit > max_logit) {
                max_logit = logit;
            }
        }

        // Compute sum(exp(logit - max_logit))
        float32_t sum_exp = 0.0f;
        for (uint64_t j = 0; j < num_classes; j++) {
            float32_t logit = logits->data[i * num_classes + j];
            sum_exp += expf(logit - max_logit);
        }

        float32_t log_sum_exp = logf(sum_exp) + max_logit;
        float32_t correct_logit = logits->data[i * num_classes + target_idx];

        // log_softmax = correct_logit - log_sum_exp
        // loss = -log_softmax
        float32_t loss = -(correct_logit - log_sum_exp);
        sum_loss += loss;
    }
    return sum_loss / (float32_t)batch_size;
}

float32_t binary_cross_entropy_loss(const Tensor *predictions, const Tensor *targets) {
    assert(predictions != NULL);
    assert(targets != NULL);
    assert(predictions->data != NULL || predictions->size == 0);
    assert(targets->data != NULL || targets->size == 0);
    assert(predictions->size == targets->size);
    assert(predictions->ndim == targets->ndim);
    for (uint64_t i = 0; i < predictions->ndim; i++) {
        assert(predictions->shape[i] == targets->shape[i]);
    }

    float32_t sum_loss = 0.0f;

    for (uint64_t i = 0; i < predictions->size; i++) {
        float32_t pred = predictions->data[i];
        float32_t target = targets->data[i];

        // Clamp prediction to avoid log(0)
        if (pred < EPSILON) {
            pred = EPSILON;
        }
        if (pred > 1.0f - EPSILON) {
            pred = 1.0f - EPSILON;
        }

        // BCE formula: -(target * log(pred) + (1 - target) * log(1 - pred))
        float32_t term1 = target * logf(pred);
        float32_t term2 = (1.0f - target) * logf(1.0f - pred);
        sum_loss += -(term1 + term2);
    }
    return sum_loss / (float32_t)predictions->size;
}
