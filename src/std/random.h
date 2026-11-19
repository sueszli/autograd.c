#pragma once

#include <stdint.h>

//
// random numbers from a 64-bit lcg, as in autograd.lean.
// state is the seed, and each call advances it.
// https://en.wikipedia.org/wiki/Linear_congruential_generator
//
// random_float:   https://numpy.org/doc/stable/reference/random/generated/numpy.random.random.html
// random_uniform: https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
// random_choice:  https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
//

// uniform in [0, 1)
double random_float(uint64_t *state);

// fills out with values uniform in [low, high)
void random_uniform(uint64_t *state, double *out, int64_t n, double low, double high);

// index i with probability p[i]
int64_t random_choice(uint64_t *state, const double *p, int64_t n);
