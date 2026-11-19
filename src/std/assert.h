#pragma once

#include "types.h"

//
// numeric asserts, named after numpy, torch and jax.
// each one aborts with a message on failure.
//
// check.allclose: https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html
// check.close:    https://pytorch.org/docs/stable/testing.html
// check.grad:     https://docs.jax.dev/en/latest/_autosummary/jax.test_util.check_grads.html
//
// differences from numpy:
// -----------------------
//     - nan never passes, where numpy treats two nans as equal
//     - check.allclose returns the max abs diff, so callers can report it
//

typedef struct {
    // passes if |actual - desired| <= atol + rtol * |desired| for every element
    f64 (*allclose)(const f64 *actual, const f64 *desired, i64 n, f64 rtol, f64 atol, const char *err_msg);

    // allclose for one scalar
    void (*close)(f64 actual, f64 expected, f64 rtol, f64 atol, const char *msg);

    // checks grad[i] against a central difference of f in x[i], for every i
    // f reads x through ctx
    void (*grad)(f64 (*f)(void *ctx), void *ctx, f64 *x, const f64 *grad, i64 n, f64 rtol, f64 atol, const char *msg);
} CheckNamespace;

// not called assert, which assert.h reserves as a macro name
extern const CheckNamespace check;
