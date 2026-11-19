#pragma once

#include "types.h"

//
// numeric asserts that abort with a message on failure.
//
// check.allclose: https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html
// check.close:    https://pytorch.org/docs/stable/testing.html
// check.grad:     https://docs.jax.dev/en/latest/_autosummary/jax.test_util.check_grads.html
//

typedef struct {
    // |actual - desired| <= atol + rtol * |desired|, nan never passes, returns the max abs diff
    f64 (*allclose)(const f64 *actual, const f64 *desired, i64 n, f64 rtol, f64 atol, const char *err_msg);

    void (*close)(f64 actual, f64 expected, f64 rtol, f64 atol, const char *msg);

    // grad against a central difference of f in each x[i]
    void (*grad)(f64 (*f)(void *ctx), void *ctx, f64 *x, const f64 *grad, i64 n, f64 rtol, f64 atol, const char *msg);
} CheckNamespace;

extern const CheckNamespace check;
