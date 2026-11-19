#include "numeric.h"
#include <math.h>
#include <stdlib.h>

void fill_uniform(double *x, int64_t n, double lo, double hi) {
    for (int64_t i = 0; i < n; i++) {
        x[i] = lo + (hi - lo) * rand() / RAND_MAX;
    }
}

double max_abs_diff(const double *a, const double *b, int64_t n) {
    double max = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (isnan(d)) {
            return INFINITY;
        }
        max = fmax(max, d);
    }
    return max;
}

double central_difference(double (*f)(void *ctx), void *ctx, double *x, double h) {
    double x0 = *x;
    *x = x0 + h;
    double up = f(ctx);
    *x = x0 - h;
    double down = f(ctx);
    *x = x0;
    return (up - down) / (2 * h);
}
