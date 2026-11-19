#include "optim.h"
#include "alloc.h"
#include <math.h>
#include <string.h>

AdamW adamw_new(Tensor **params, int64_t n) {
    AdamW opt = {.params = params, .n = n, .m = fba_alloc(&perm_mem, n * (int64_t)sizeof(double *)), .v = fba_alloc(&perm_mem, n * (int64_t)sizeof(double *))};
    for (int64_t i = 0; i < n; i++) {
        int64_t bytes = params[i]->rows * params[i]->cols * (int64_t)sizeof(double);
        opt.m[i] = fba_alloc(&perm_mem, bytes);
        opt.v[i] = fba_alloc(&perm_mem, bytes);
    }
    return opt;
}

// betas and eps match the python reference. sqrt instead of pow(x, 0.5): pow was half the training time
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
            p->data[j] = p->data[j] - lr_scaled * m[j] / (sqrt(v[j] * inv_bias2) + eps);
        }
        if (p->grad != NULL) {
            memset(p->grad, 0, (size_t)(p->rows * p->cols) * sizeof(double));
        }
    }
}
