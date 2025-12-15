#include "losses_backward.h"
#include "tensor.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

Tensor *mse_loss_backward(const Tensor *predictions, const Tensor *targets) {
    assert(predictions != NULL);
    assert(targets != NULL);
    assert(predictions->data != NULL || predictions->size == 0);
    assert(targets->data != NULL || targets->size == 0);
    assert(predictions->size == targets->size);
    assert(predictions->ndim == targets->ndim);

    for (uint64_t i = 0; i < predictions->ndim; i++) {
        assert(predictions->shape[i] == targets->shape[i]);
    }

    Tensor *grad = tensor_create(NULL, predictions->shape, predictions->ndim, false);
    if (predictions->size == 0) {
        return grad;
    }

    float32_t inv_n = 2.0f / (float32_t)predictions->size;
    for (uint64_t i = 0; i < predictions->size; i++) {
        float32_t diff = predictions->data[i] - targets->data[i];
        grad->data[i] = inv_n * diff;
    }

    return grad;
}

Tensor *cross_entropy_loss_backward(const Tensor *logits, const Tensor *targets) {
    assert(logits != NULL);
    assert(targets != NULL);
    assert(logits->data != NULL || logits->size == 0);
    assert(targets->data != NULL || targets->size == 0);

    assert(logits->ndim == 2);
    assert(targets->ndim == 1 || (targets->ndim == 2 && targets->shape[1] == 1));
    assert(logits->shape[0] == targets->shape[0]);

    uint64_t batch_size = logits->shape[0];
    uint64_t num_classes = logits->shape[1];

    Tensor *grad = tensor_create(NULL, logits->shape, logits->ndim, false);
    if (batch_size == 0 || num_classes == 0) {
        return grad;
    }

    float32_t inv_batch = 1.0f / (float32_t)batch_size;

    for (uint64_t i = 0; i < batch_size; i++) {
        float32_t target_float = targets->data[i];
        assert(target_float >= 0.0f && target_float < (float32_t)num_classes);
        uint64_t target_idx = (uint64_t)target_float;

        // for numerical stability, compute softmax via log-sum-exp trick
        float32_t max_logit = -FLT_MAX;
        for (uint64_t j = 0; j < num_classes; j++) {
            float32_t logit = logits->data[i * num_classes + j];
            if (logit > max_logit) {
                max_logit = logit;
            }
        }

        float32_t sum_exp = 0.0f;
        for (uint64_t j = 0; j < num_classes; j++) {
            float32_t logit = logits->data[i * num_classes + j];
            sum_exp += expf(logit - max_logit);
        }

        if (sum_exp < 1.0f) {
            sum_exp = 1.0f;
        }

        // softmax probabilities
        for (uint64_t j = 0; j < num_classes; j++) {
            float32_t logit = logits->data[i * num_classes + j];
            float32_t prob = expf(logit - max_logit) / sum_exp;
            float32_t indicator = (j == target_idx) ? 1.0f : 0.0f;
            grad->data[i * num_classes + j] = (prob - indicator) * inv_batch;
        }
    }

    return grad;
}

#define EPSILON 1e-7f

Tensor *binary_cross_entropy_loss_backward(const Tensor *predictions, const Tensor *targets) {
    assert(predictions != NULL);
    assert(targets != NULL);
    assert(predictions->data != NULL || predictions->size == 0);
    assert(targets->data != NULL || targets->size == 0);
    assert(predictions->size == targets->size);
    assert(predictions->ndim == targets->ndim);

    for (uint64_t i = 0; i < predictions->ndim; i++) {
        assert(predictions->shape[i] == targets->shape[i]);
    }

    Tensor *grad = tensor_create(NULL, predictions->shape, predictions->ndim, false);
    if (predictions->size == 0) {
        return grad;
    }

    float32_t inv_n = 1.0f / (float32_t)predictions->size;

    for (uint64_t i = 0; i < predictions->size; i++) {
        float32_t p = predictions->data[i];
        float32_t t = targets->data[i];

        // clamp p into (0,1) for numerical stability, mirroring forward
        if (p < EPSILON) {
            p = EPSILON;
        }
        if (p > 1.0f - EPSILON) {
            p = 1.0f - EPSILON;
        }

        float32_t denom = p * (1.0f - p);
        if (denom < EPSILON) {
            denom = EPSILON;
        }

        grad->data[i] = ((p - t) / denom) * inv_n;
    }

    return grad;
}
