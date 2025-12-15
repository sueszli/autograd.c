#include "optimizers.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

//
// base implementation
//

void optimizer_zero_grad(Optimizer *opt) {
    assert(opt != NULL);
    for (size_t i = 0; i < opt->param_count; ++i) {
        Tensor *param = opt->params[i];
        if (param->grad != NULL) {
            tensor_free(param->grad);
            param->grad = NULL;
        }
    }
}

void optimizer_step(Optimizer *opt) {
    assert(opt != NULL);
    opt->step(opt);
}

void optimizer_free(Optimizer *opt) {
    if (opt != NULL) {
        opt->free(opt);
    }
}

//
// sgd
//

typedef struct {
    Optimizer base;
    float32_t lr;
    float32_t momentum;
    float32_t weight_decay;
    float32_t **momentum_buffers; // array of pointers to float arrays (same size as param data)
} SGD;

static void sgd_free(Optimizer *opt) {
    SGD *sgd = (SGD *)opt;
    if (sgd->momentum_buffers) {
        for (size_t i = 0; i < opt->param_count; ++i) {
            if (sgd->momentum_buffers[i]) {
                free(sgd->momentum_buffers[i]);
            }
        }
        free(sgd->momentum_buffers);
    }
    if (opt->params) {
        free(opt->params);
    }
    free(sgd);
}

static void sgd_ensure_buffer(SGD *sgd, size_t param_idx, size_t elem_count) {
    if (sgd->momentum_buffers[param_idx] == NULL) {
        // use calloc for zero initialization
        float32_t *buf = calloc(elem_count, sizeof(float32_t));
        assert(buf != NULL);
        sgd->momentum_buffers[param_idx] = buf;
    }
}

static void sgd_step(Optimizer *opt) {
    SGD *sgd = (SGD *)opt;
    opt->step_count++;

    for (size_t i = 0; i < opt->param_count; ++i) {
        Tensor *param = opt->params[i];
        if (param->grad == NULL) {
            continue;
        }

        float32_t *p_data = param->data;
        const float32_t *g_data = param->grad->data;
        // Rename size to elem_count for clarity (CONTRIBUTING.md)
        const size_t elem_count = param->size;

        // allocate momentum buffer if needed
        if (sgd->momentum != 0.0f) {
            sgd_ensure_buffer(sgd, i, elem_count);
        }

        float32_t *m_buf = sgd->momentum_buffers[i];

        // optimize: lift constants out of loop
        const float32_t lr = sgd->lr;
        const float32_t momentum = sgd->momentum;
        const float32_t weight_decay = sgd->weight_decay;

        // Data-oriented: process arrays
        for (size_t j = 0; j < elem_count; ++j) {
            float32_t g = g_data[j];

            // 1. Weight decay (L2 penalty)
            if (weight_decay != 0.0f) {
                g += weight_decay * p_data[j];
            }

            // 2. Momentum
            if (momentum != 0.0f) {
                // v = momentum * v_prev + g
                m_buf[j] = momentum * m_buf[j] + g;
                // update gradient to be the velocity
                g = m_buf[j];
            }

            // 3. Update parameter
            p_data[j] -= lr * g;
        }
    }
}

Optimizer *optimizer_sgd_create(Tensor **params, size_t count, float32_t lr, float32_t momentum, float32_t weight_decay) {
    assert(params != NULL);
    assert(count > 0);

    Optimizer *opt = calloc(1, sizeof(SGD));
    assert(opt != NULL);
    SGD *sgd = (SGD *)opt;

    sgd->base.param_count = count;
    // Copy the params array so validation/integrity is kept
    sgd->base.params = calloc(count, sizeof(Tensor *));
    assert(sgd->base.params != NULL);
    for (size_t i = 0; i < count; ++i) {
        // Assert every param requires grad check? Reference says check requires_grad
        assert(params[i] != NULL);
        assert(params[i]->requires_grad && "All optimized tensors must require grad");
        sgd->base.params[i] = params[i];
    }

    sgd->base.step = sgd_step;
    sgd->base.free = sgd_free;
    sgd->base.step_count = 0;

    sgd->lr = lr;
    sgd->momentum = momentum;
    sgd->weight_decay = weight_decay;

    sgd->momentum_buffers = calloc(count, sizeof(float32_t *));
    assert(sgd->momentum_buffers != NULL);

    return opt;
}

//
// adam
//

typedef struct {
    Optimizer base;
    float32_t lr;
    float32_t beta1;
    float32_t beta2;
    float32_t eps;
    float32_t weight_decay;
    float32_t **m_buffers; // First moment
    float32_t **v_buffers; // Second moment
} Adam;

static void adam_free(Optimizer *opt) {
    Adam *adam = (Adam *)opt;
    if (adam->m_buffers) {
        for (size_t i = 0; i < opt->param_count; ++i) {
            if (adam->m_buffers[i])
                free(adam->m_buffers[i]);
        }
        free(adam->m_buffers);
    }
    if (adam->v_buffers) {
        for (size_t i = 0; i < opt->param_count; ++i) {
            if (adam->v_buffers[i])
                free(adam->v_buffers[i]);
        }
        free(adam->v_buffers);
    }
    if (opt->params) {
        free(opt->params);
    }
    free(adam);
}

static void adam_ensure_buffers(Adam *adam, size_t param_idx, size_t elem_count) {
    if (adam->m_buffers[param_idx] == NULL) {
        adam->m_buffers[param_idx] = calloc(elem_count, sizeof(float32_t));
        assert(adam->m_buffers[param_idx] != NULL);
        adam->v_buffers[param_idx] = calloc(elem_count, sizeof(float32_t));
        assert(adam->v_buffers[param_idx] != NULL);
    }
}

static void adam_step_impl(Optimizer *opt, bool is_adamw) {
    Adam *adam = (Adam *)opt;
    opt->step_count++;

    // cache constants for loop efficiency
    const float32_t beta1 = adam->beta1;
    const float32_t beta2 = adam->beta2;
    const float32_t eps = adam->eps;
    const float32_t weight_decay = adam->weight_decay;
    const float32_t lr = adam->lr;

    // pre-compute bias corrections (constant across all parameters)
    // 1 - beta^t
    const float32_t bias_correction1 = 1.0f - (float32_t)pow(beta1, (double)opt->step_count);
    const float32_t bias_correction2 = 1.0f - (float32_t)pow(beta2, (double)opt->step_count);

    for (size_t i = 0; i < opt->param_count; ++i) {
        Tensor *param = opt->params[i];
        if (param->grad == NULL) {
            continue;
        }

        float32_t *p_data = param->data;
        const float32_t *g_data = param->grad->data;
        const size_t elem_count = param->size; // rename size -> elem_count

        // ensure internal state buffers exist
        adam_ensure_buffers(adam, i, elem_count);

        float32_t *m = adam->m_buffers[i];
        float32_t *v = adam->v_buffers[i];

        // hot loop: process ensure strict alignment / SIMD friendly if possible
        for (size_t j = 0; j < elem_count; ++j) {
            float32_t g = g_data[j];

            if (!is_adamw) {
                // Adam: Add weight decay to gradient (L2 regularization equivalent)
                if (weight_decay != 0.0f) {
                    g += weight_decay * p_data[j];
                }
            }

            // 1. Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
            m[j] = beta1 * m[j] + (1.0f - beta1) * g;

            // 2. Update biased second moment estimate: v = beta2 * v + (1 - beta2) * g^2
            v[j] = beta2 * v[j] + (1.0f - beta2) * (g * g);

            // 3. Compute bias-corrected moments
            float32_t m_hat = m[j] / bias_correction1;
            float32_t v_hat = v[j] / bias_correction2;

            // 4. Update parameter
            p_data[j] -= lr * m_hat / (sqrtf(v_hat) + eps);

            if (is_adamw) {
                // AdamW: Decay weights directly (decoupled weight decay)
                // P_new = P_old - lr * (weight_decay * P_old + other_terms)
                if (weight_decay != 0.0f) {
                    p_data[j] *= (1.0f - lr * weight_decay);
                }
            }
        }
    }
}

static void adam_step(Optimizer *opt) { adam_step_impl(opt, false); }

static void adamw_step(Optimizer *opt) { adam_step_impl(opt, true); }

Optimizer *optimizer_adam_create(Tensor **params, size_t count, float32_t lr, float32_t beta1, float32_t beta2, float32_t eps, float32_t weight_decay) {
    assert(params != NULL);
    assert(count > 0);

    Optimizer *opt = calloc(1, sizeof(Adam));
    assert(opt != NULL);
    Adam *adam = (Adam *)opt;

    adam->base.param_count = count;
    adam->base.params = calloc(count, sizeof(Tensor *));
    assert(adam->base.params != NULL);
    for (size_t i = 0; i < count; ++i) {
        assert(params[i] != NULL);
        assert(params[i]->requires_grad && "All optimized tensors must require grad");
        adam->base.params[i] = params[i];
    }

    adam->base.step = adam_step;
    adam->base.free = adam_free;
    adam->base.step_count = 0;

    adam->lr = lr;
    adam->beta1 = beta1;
    adam->beta2 = beta2;
    adam->eps = eps;
    adam->weight_decay = weight_decay;

    adam->m_buffers = calloc(count, sizeof(float32_t *));
    assert(adam->m_buffers != NULL);
    adam->v_buffers = calloc(count, sizeof(float32_t *));
    assert(adam->v_buffers != NULL);

    return opt;
}

Optimizer *optimizer_adamw_create(Tensor **params, size_t count, float32_t lr, float32_t beta1, float32_t beta2, float32_t eps, float32_t weight_decay) {
    assert(params != NULL);
    assert(count > 0);

    Optimizer *opt = calloc(1, sizeof(Adam));
    assert(opt != NULL);
    Adam *adam = (Adam *)opt;

    adam->base.param_count = count;
    adam->base.params = calloc(count, sizeof(Tensor *));
    assert(adam->base.params != NULL);
    for (size_t i = 0; i < count; ++i) {
        assert(params[i] != NULL);
        assert(params[i]->requires_grad && "All optimized tensors must require grad");
        adam->base.params[i] = params[i];
    }

    adam->base.step = adamw_step; // Different step function
    adam->base.free = adam_free;
    adam->base.step_count = 0;

    adam->lr = lr;
    adam->beta1 = beta1;
    adam->beta2 = beta2;
    adam->eps = eps;
    adam->weight_decay = weight_decay;

    adam->m_buffers = calloc(count, sizeof(float32_t *));
    assert(adam->m_buffers != NULL);
    adam->v_buffers = calloc(count, sizeof(float32_t *));
    assert(adam->v_buffers != NULL);

    return opt;
}
