#pragma once

#include <stdint.h>

//
// test helpers, named after numpy and torch
//
// random_uniform:     https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
// assert_allclose:    https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html
// assert_close:       https://pytorch.org/docs/stable/testing.html
// central_difference: https://en.wikipedia.org/wiki/Finite_difference
//
// differences from numpy:
// -----------------------
//     - nan never passes. numpy treats two nans as equal
//     - assert_allclose returns the max abs diff, so callers can report it
//

// fills out with values in [low, high]. uses rand(), so srand seeds it
void random_uniform(double *out, int64_t n, double low, double high);

// aborts with err_msg unless |actual - desired| <= atol + rtol * |desired| for every element
double assert_allclose(const double *actual, const double *desired, int64_t n, double rtol, double atol, const char *err_msg);

// assert_allclose for one scalar
void assert_close(double actual, double expected, double rtol, double atol, const char *msg);

// (f(x + h) - f(x - h)) / 2h. nudges *x in place, then restores it
double central_difference(double (*f)(void *ctx), void *ctx, double *x, double h);
