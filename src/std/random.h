#pragma once

#include "types.h"

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
f64 random_float(u64 *state);

// fills out with values uniform in [low, high)
void random_uniform(u64 *state, f64 *out, i64 n, f64 low, f64 high);

// index i with probability p[i]
i64 random_choice(u64 *state, const f64 *p, i64 n);
