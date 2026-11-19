#pragma once

#include <stdint.h>

//
// numeric asserts, named after numpy, torch and jax.
// each one aborts with a message on failure.
//
// assert_allclose: https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html
// assert_close:    https://pytorch.org/docs/stable/testing.html
// assert_grad:     https://docs.jax.dev/en/latest/_autosummary/jax.test_util.check_grads.html
//
// differences from numpy:
// -----------------------
//     - nan never passes, where numpy treats two nans as equal
//     - assert_allclose returns the max abs diff, so callers can report it
//

// passes if |actual - desired| <= atol + rtol * |desired| for every element
double assert_allclose(const double *actual, const double *desired, int64_t n, double rtol, double atol, const char *err_msg);

// assert_allclose for one scalar
void assert_close(double actual, double expected, double rtol, double atol, const char *msg);

// checks grad[i] against a central difference of f in x[i], for every i
// f reads x through ctx
void assert_grad(double (*f)(void *ctx), void *ctx, double *x, const double *grad, int64_t n, double rtol, double atol, const char *msg);
