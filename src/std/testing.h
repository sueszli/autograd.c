#pragma once

#include <stdint.h>

// numpy.random.uniform: fills out with values in [low, high] from rand(), so srand seeds it
void random_uniform(double *out, int64_t n, double low, double high);

// numpy.testing.assert_allclose: aborts with err_msg unless |actual - desired| <= atol + rtol * |desired| everywhere.
// unlike numpy, nan never passes. returns the max abs diff
double assert_allclose(const double *actual, const double *desired, int64_t n, double rtol, double atol, const char *err_msg);

// torch.testing.assert_close for one scalar, same check as assert_allclose
void assert_close(double actual, double expected, double rtol, double atol, const char *msg);

// (f(x + h) - f(x - h)) / 2h, nudging *x in place and restoring it
double central_difference(double (*f)(void *ctx), void *ctx, double *x, double h);
