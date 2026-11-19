#pragma once

#include <stdint.h>

// uniform in [lo, hi] from rand(), so srand seeds it
void fill_uniform(double *x, int64_t n, double lo, double hi);

// largest |a[i] - b[i]|
double max_abs_diff(const double *a, const double *b, int64_t n);

// (f(x + h) - f(x - h)) / 2h, nudging *x in place and restoring it
double central_difference(double (*f)(void *ctx), void *ctx, double *x, double h);
