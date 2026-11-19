#include "testing.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void random_uniform(double *out, int64_t n, double low, double high) {
    // low + (high - low) * u, with u uniform in [0, 1]
    for (int64_t i = 0; i < n; i++) {
        out[i] = low + (high - low) * rand() / RAND_MAX;
    }
}

double assert_allclose(const double *actual, const double *desired, int64_t n, double rtol, double atol, const char *err_msg) {
    // !(diff <= tol) instead of diff > tol, so a nan fails
    double max = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double diff = fabs(actual[i] - desired[i]);
        if (!(diff <= atol + rtol * fabs(desired[i]))) {
            fprintf(stderr, "not close: %s[%ld] is %.17g, expected %.17g, abs diff %g\n", err_msg, (long)i, actual[i], desired[i], diff);
            abort();
        }
        max = fmax(max, diff);
    }
    return max;
}

void assert_close(double actual, double expected, double rtol, double atol, const char *msg) {
    // a scalar is a 1-element array
    assert_allclose(&actual, &expected, 1, rtol, atol, msg);
}

double central_difference(double (*f)(void *ctx), void *ctx, double *x, double h) {
    // symmetric difference, error shrinks with h^2
    double x0 = *x;
    *x = x0 + h;
    double up = f(ctx);
    *x = x0 - h;
    double down = f(ctx);
    *x = x0;
    return (up - down) / (2 * h);
}
