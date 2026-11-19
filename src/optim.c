#include "optim.h"
#include <assert.h>
#include <math.h>
#include <string.h>

AdamW adamw_new(Tensor **params, int64_t n) {
    AdamW opt = {.params = params, .n = n, .m = calloc((size_t)n, sizeof(double *)), .v = calloc((size_t)n, sizeof(double *))};
    assert(opt.m != NULL && opt.v != NULL && "calloc failed");
    for (int64_t i = 0; i < n; i++) {
        size_t size = (size_t)(params[i]->rows * params[i]->cols);
        opt.m[i] = calloc(size, sizeof(double));
        opt.v[i] = calloc(size, sizeof(double));
        assert(opt.m[i] != NULL && opt.v[i] != NULL && "calloc failed");
    }
    return opt;
}

// betas and eps match the python reference, and the expression order matches autograd.lean
void adamw_step(AdamW *opt, int64_t step, int64_t num_steps) {
    const double beta1 = 0.85, beta2 = 0.99, eps = 1e-8;
    double lr = fmax(0.01 * (1.0 - (double)(step - 1) / (double)num_steps), 0.0);
    double inv_bias1 = 1.0 / (1.0 - pow(beta1, (double)step));
    double inv_bias2 = 1.0 / (1.0 - pow(beta2, (double)step));
    double lr_scaled = lr * inv_bias1;
    for (int64_t i = 0; i < opt->n; i++) {
        Tensor *p = opt->params[i];
        double *m = opt->m[i], *v = opt->v[i];
        for (int64_t j = 0; j < p->rows * p->cols; j++) {
            double g = p->grad != NULL ? p->grad[j] : 0.0;
            m[j] = beta1 * m[j] + (1.0 - beta1) * g;
            v[j] = beta2 * v[j] + (1.0 - beta2) * g * g;
            p->data[j] = p->data[j] - lr_scaled * m[j] / (pow(v[j] * inv_bias2, 0.5) + eps);
        }
        if (p->grad != NULL) {
            memset(p->grad, 0, (size_t)(p->rows * p->cols) * sizeof(double));
        }
    }
}

void adamw_free(AdamW *opt) {
    for (int64_t i = 0; i < opt->n; i++) {
        free(opt->m[i]);
        free(opt->v[i]);
    }
    free(opt->m);
    free(opt->v);
}
