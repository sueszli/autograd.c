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
        if (param->grad == NULL) {
            continue;
        }
        tensor_free(param->grad);
        param->grad = NULL;
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
    if (sgd->momentum_buffers[param_idx] != NULL) {
        return;
    }
    float32_t *buf = calloc(elem_count, sizeof(float32_t));
    assert(buf != NULL);
    sgd->momentum_buffers[param_idx] = buf;
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
        const size_t elem_count = param->size;

        if (sgd->momentum != 0.0f) {
            sgd_ensure_buffer(sgd, i, elem_count);
        }

        float32_t *m_buf = sgd->momentum_buffers[i];
        const float32_t lr = sgd->lr;
        const float32_t momentum = sgd->momentum;
        const float32_t weight_decay = sgd->weight_decay;

        for (size_t j = 0; j < elem_count; ++j) {
            float32_t g = g_data[j];

            // weight decay (L2 penalty)
            if (weight_decay != 0.0f) {
                g += weight_decay * p_data[j];
            }

            // momentum
            if (momentum != 0.0f) {
                // v = momentum * v_prev + g
                m_buf[j] = momentum * m_buf[j] + g;
                // update gradient to be the velocity
                g = m_buf[j];
            }

            // update parameter
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
    // copy the params array so validation/integrity is kept
    sgd->base.params = calloc(count, sizeof(Tensor *));
    assert(sgd->base.params != NULL);
    for (size_t i = 0; i < count; ++i) {
        assert(params[i] != NULL);
        assert(params[i]->requires_grad && "all optimized tensors must require grad");
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
    float32_t **m_buffers; // first moment
    float32_t **v_buffers; // second moment
} Adam;

static void adam_free(Optimizer *opt) {
    Adam *adam = (Adam *)opt;
    if (adam->m_buffers) {
        for (size_t i = 0; i < opt->param_count; ++i) {
            if (adam->m_buffers[i]) {
                free(adam->m_buffers[i]);
            }
        }
        free(adam->m_buffers);
    }
    if (adam->v_buffers) {
        for (size_t i = 0; i < opt->param_count; ++i) {
            if (adam->v_buffers[i]) {
                free(adam->v_buffers[i]);
            }
        }
        free(adam->v_buffers);
    }
    if (opt->params) {
        free(opt->params);
    }
    free(adam);
}

static void adam_ensure_buffers(Adam *adam, size_t param_idx, size_t elem_count) {
    if (adam->m_buffers[param_idx] != NULL) {
        return;
    }
    adam->m_buffers[param_idx] = calloc(elem_count, sizeof(float32_t));
    assert(adam->m_buffers[param_idx] != NULL);
    adam->v_buffers[param_idx] = calloc(elem_count, sizeof(float32_t));
    assert(adam->v_buffers[param_idx] != NULL);
}

static void adam_step_impl(Optimizer *opt, bool is_adamw) {
    Adam *adam = (Adam *)opt;
    opt->step_count++;

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
        const size_t elem_count = param->size;

        // ensure internal state buffers exist
        adam_ensure_buffers(adam, i, elem_count);

        float32_t *m = adam->m_buffers[i];
        float32_t *v = adam->v_buffers[i];

        for (size_t j = 0; j < elem_count; ++j) {
            float32_t g = g_data[j];

            if (!is_adamw) {
                // adam: Add weight decay to gradient (L2 regularization equivalent)
                if (weight_decay != 0.0f) {
                    g += weight_decay * p_data[j];
                }
            }

            // update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
            m[j] = beta1 * m[j] + (1.0f - beta1) * g;

            // update biased second moment estimate: v = beta2 * v + (1 - beta2) * g^2
            v[j] = beta2 * v[j] + (1.0f - beta2) * (g * g);

            // compute bias-corrected moments
            float32_t m_hat = m[j] / bias_correction1;
            float32_t v_hat = v[j] / bias_correction2;

            // update parameter
            p_data[j] -= lr * m_hat / (sqrtf(v_hat) + eps);

            if (is_adamw) {
                // AdamW: decay weights directly (decoupled weight decay)
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

static Optimizer *adam_create_internal(Tensor **params, size_t count, float32_t lr, float32_t beta1, float32_t beta2, float32_t eps, float32_t weight_decay, void (*step_fn)(Optimizer *)) {
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
        assert(params[i]->requires_grad && "all optimized tensors must require grad");
        adam->base.params[i] = params[i];
    }

    adam->base.step = step_fn; // this is the only difference between adam and adamw
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

Optimizer *optimizer_adam_create(Tensor **params, size_t count, float32_t lr, float32_t beta1, float32_t beta2, float32_t eps, float32_t weight_decay) { return adam_create_internal(params, count, lr, beta1, beta2, eps, weight_decay, adam_step); }

Optimizer *optimizer_adamw_create(Tensor **params, size_t count, float32_t lr, float32_t beta1, float32_t beta2, float32_t eps, float32_t weight_decay) { return adam_create_internal(params, count, lr, beta1, beta2, eps, weight_decay, adamw_step); }
