#include "random.h"

static f64 random_random(u64 *state) {
    // knuth's mmix constants, then the top 53 bits as a f64
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (f64)(*state >> 11) / 9007199254740992.0;
}

static void random_uniform(u64 *state, f64 *out, i64 n, f64 low, f64 high) {
    // low + (high - low) * u
    for (i64 i = 0; i < n; i++) {
        out[i] = low + (high - low) * random_random(state);
    }
}

static i64 random_choice(u64 *state, const f64 *p, i64 n) {
    // walk the cumulative sum until it passes u, the last index catches rounding
    f64 u = random_random(state);
    for (i64 i = 0; i < n - 1; i++) {
        u -= p[i];
        if (u < 0.0) {
            return i;
        }
    }
    return n - 1;
}

const RngNamespace rng = {.random = random_random, .uniform = random_uniform, .choice = random_choice};
