#pragma once

#include "types.h"

//
// random numbers from a 64-bit lcg, as in autograd.lean.
// state is the seed, and each call advances it.
// https://en.wikipedia.org/wiki/Linear_congruential_generator
//
// rng.random:  https://numpy.org/doc/stable/reference/random/generated/numpy.random.random.html
// rng.uniform: https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
// rng.choice:  https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
//

typedef struct {
    f64 (*random)(u64 *state);                                       // uniform in [0, 1)
    void (*uniform)(u64 *state, f64 *out, i64 n, f64 low, f64 high); // fills out with values uniform in [low, high)
    i64 (*choice)(u64 *state, const f64 *p, i64 n);                  // index i with probability p[i]
} RngNamespace;

// not called random, which posix already declares in stdlib.h
extern const RngNamespace rng;
