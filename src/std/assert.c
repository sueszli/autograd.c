#include "assert.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

void assert_grad(double (*f)(void *ctx), void *ctx, double *x, const double *grad, int64_t n, double rtol, double atol, const char *msg) {
    // (f(x + h) - f(x - h)) / 2h, its error shrinks with h^2
    const double h = 1e-6;
    for (int64_t i = 0; i < n; i++) {
        double x0 = x[i];
        x[i] = x0 + h;
        double up = f(ctx);
        x[i] = x0 - h;
        double down = f(ctx);
        x[i] = x0;
        assert_close(grad[i], (up - down) / (2 * h), rtol, atol, msg);
    }
}
