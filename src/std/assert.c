#include "assert.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static f64 assert_allclose(const f64 *actual, const f64 *desired, i64 n, f64 rtol, f64 atol, const char *err_msg) {
    // !(diff <= tol) instead of diff > tol, so a nan fails
    f64 max = 0.0;
    for (i64 i = 0; i < n; i++) {
        f64 diff = fabs(actual[i] - desired[i]);
        if (!(diff <= atol + rtol * fabs(desired[i]))) {
            fprintf(stderr, "not close: %s[%ld] is %.17g, expected %.17g, abs diff %g\n", err_msg, (long)i, actual[i], desired[i], diff);
            abort();
        }
        max = fmax(max, diff);
    }
    return max;
}

static void assert_close(f64 actual, f64 expected, f64 rtol, f64 atol, const char *msg) {
    // a scalar is a 1-element array
    assert_allclose(&actual, &expected, 1, rtol, atol, msg);
}

static void assert_grad(f64 (*f)(void *ctx), void *ctx, f64 *x, const f64 *grad, i64 n, f64 rtol, f64 atol, const char *msg) {
    // (f(x + h) - f(x - h)) / 2h, its error shrinks with h^2
    const f64 h = 1e-6;
    for (i64 i = 0; i < n; i++) {
        f64 x0 = x[i];
        x[i] = x0 + h;
        f64 up = f(ctx);
        x[i] = x0 - h;
        f64 down = f(ctx);
        x[i] = x0;
        assert_close(grad[i], (up - down) / (2 * h), rtol, atol, msg);
    }
}

const CheckNamespace check = {.allclose = assert_allclose, .close = assert_close, .grad = assert_grad};
