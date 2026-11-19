#include "optim.h"
#include "tensor.h"
#include <math.h>
#include <string.h>

AdamW adamw_new(Tensor **params, i64 n) {
    AdamW opt = {.params = params, .n = n, .m = perm_alloc(n * (i64)sizeof(f64 *)), .v = perm_alloc(n * (i64)sizeof(f64 *))};
    for (i64 i = 0; i < n; i++) {
        i64 bytes = params[i]->size * (i64)sizeof(f64);
        opt.m[i] = perm_alloc(bytes);
        opt.v[i] = perm_alloc(bytes);
    }
    return opt;
}

// betas and eps match the python reference
// sqrt instead of pow(x, 0.5), pow was half the training time
void adamw_step(AdamW *opt, i64 step, i64 num_steps) {
    const f64 beta1 = 0.85, beta2 = 0.99, eps = 1e-8;
    f64 lr = 0.01 * (1.0 - (f64)(step - 1) / (f64)num_steps);
    f64 inv_bias1 = 1.0 / (1.0 - pow(beta1, (f64)step));
    f64 inv_bias2 = 1.0 / (1.0 - pow(beta2, (f64)step));
    f64 lr_scaled = lr * inv_bias1;
    for (i64 i = 0; i < opt->n; i++) {
        Tensor *p = opt->params[i];
        f64 *m = opt->m[i], *v = opt->v[i];
        for (i64 j = 0; j < p->size; j++) {
            f64 g = p->grad[j];
            m[j] = beta1 * m[j] + (1.0 - beta1) * g;
            v[j] = beta2 * v[j] + (1.0 - beta2) * g * g;
            p->data[j] = p->data[j] - lr_scaled * m[j] / (sqrt(v[j] * inv_bias2) + eps);
        }
        memset(p->grad, 0, (usize)p->size * sizeof(f64));
    }
}
