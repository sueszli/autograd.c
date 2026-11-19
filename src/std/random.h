#pragma once

#include "types.h"

//
// numpy.random on a 64-bit lcg, each call advances the seed in state.
// https://numpy.org/doc/stable/reference/random/legacy.html
//

typedef struct {
    f64 (*random)(u64 *state);                                       // uniform in [0, 1)
    void (*uniform)(u64 *state, f64 *out, i64 n, f64 low, f64 high); // fills out with values uniform in [low, high)
    i64 (*choice)(u64 *state, const f64 *p, i64 n);                  // i with probability p[i]
} RngNamespace;

extern const RngNamespace rng;
