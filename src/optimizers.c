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
    // params array is just a reference, caller owns the tensors.
    // but the array itself 'opt->params' was allocated by us in create?
    // The base struct has 'Tensor **params'.
    // We need to manage the allocation of the base struct parts too?
    // In create function we will allocate 'params' copy.
    if (opt->params) {
        free(opt->params);
    }
    free(sgd);
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
        size_t size = param->size;

        // Ensure momentum buffer exists if needed
        if (sgd->momentum != 0.0f) {
            if (sgd->momentum_buffers[i] == NULL) {
                // allocations must be guarded / asserted
                float32_t *buf = calloc(size, sizeof(float32_t));
                assert(buf != NULL);
                sgd->momentum_buffers[i] = buf;
            }
        }

        float32_t *m_buf = sgd->momentum_buffers[i];

        // We can vectorize this loop or rely on compiler O2/O3
        for (size_t j = 0; j < size; ++j) {
            float32_t g = g_data[j];

            // Weight decay
            if (sgd->weight_decay != 0.0f) {
                g += sgd->weight_decay * p_data[j];
            }

            // Momentum
            if (sgd->momentum != 0.0f) {
                // v = momentum * v_prev + g
                m_buf[j] = sgd->momentum * m_buf[j] + g;
                g = m_buf[j];
            }

            // Update
            p_data[j] -= sgd->lr * g;
        }
    }
}

Optimizer *optimizer_sgd_create(Tensor **params, size_t count, float32_t lr, float32_t momentum, float32_t weight_decay) {
    assert(params != NULL);
    assert(count > 0);

    SGD *sgd = calloc(1, sizeof(SGD));
    assert(sgd != NULL);

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

    Optimizer *opt = (Optimizer *)sgd;
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

static void adam_step_impl(Optimizer *opt, bool is_adamw) {
    Adam *adam = (Adam *)opt;
    opt->step_count++;

    // Precompute bias corrections
    // pow returns double, cast to float32_t
    float32_t bias_correction1 = 1.0f - (float32_t)pow(adam->beta1, (double)opt->step_count);
    float32_t bias_correction2 = 1.0f - (float32_t)pow(adam->beta2, (double)opt->step_count);

    for (size_t i = 0; i < opt->param_count; ++i) {
        Tensor *param = opt->params[i];
        if (param->grad == NULL) {
            continue;
        }

        float32_t *p_data = param->data;
        const float32_t *g_data = param->grad->data;
        size_t size = param->size;

        if (adam->m_buffers[i] == NULL) {
            adam->m_buffers[i] = calloc(size, sizeof(float32_t));
            assert(adam->m_buffers[i] != NULL);
            adam->v_buffers[i] = calloc(size, sizeof(float32_t));
            assert(adam->v_buffers[i] != NULL);
        }

        float32_t *m = adam->m_buffers[i];
        float32_t *v = adam->v_buffers[i];

        for (size_t j = 0; j < size; ++j) {
            float32_t g = g_data[j];

            if (!is_adamw) {
                // Adam: Add weight decay to gradient
                if (adam->weight_decay != 0.0f) {
                    g += adam->weight_decay * p_data[j];
                }
            }

            // Update biased first moment estimate
            m[j] = adam->beta1 * m[j] + (1.0f - adam->beta1) * g;

            // Update biased second moment estimate
            v[j] = adam->beta2 * v[j] + (1.0f - adam->beta2) * (g * g);

            // Compute bias-corrected moments
            float32_t m_hat = m[j] / bias_correction1;
            float32_t v_hat = v[j] / bias_correction2;

            // Update parameter
            p_data[j] -= adam->lr * m_hat / (sqrtf(v_hat) + adam->eps);

            if (is_adamw) {
                // AdamW: Decay weights directly
                if (adam->weight_decay != 0.0f) {
                    p_data[j] *= (1.0f - adam->lr * adam->weight_decay);
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

    Adam *adam = calloc(1, sizeof(Adam));
    assert(adam != NULL);

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

    Optimizer *opt = (Optimizer *)adam;
    return opt;
}

Optimizer *optimizer_adamw_create(Tensor **params, size_t count, float32_t lr, float32_t beta1, float32_t beta2, float32_t eps, float32_t weight_decay) {
    // Re-use initialization logic? Or simple copy paste.
    // It's cleaner to just do it again or have a helper but we need specific class/step func.
    // Let's copy-paste and change step func, simple enough.

    assert(params != NULL);
    assert(count > 0);

    Adam *adam = calloc(1, sizeof(Adam));
    assert(adam != NULL);

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

    Optimizer *opt = (Optimizer *)adam;
    return opt;
}
