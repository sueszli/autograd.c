#include "random.h"

double random_float(uint64_t *state) {
    // knuth's mmix constants, then the top 53 bits as a double
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(*state >> 11) / 9007199254740992.0;
}

void random_uniform(uint64_t *state, double *out, int64_t n, double low, double high) {
    // low + (high - low) * u
    for (int64_t i = 0; i < n; i++) {
        out[i] = low + (high - low) * random_float(state);
    }
}

int64_t random_choice(uint64_t *state, const double *p, int64_t n) {
    // walk the cumulative sum until it passes u, the last index catches rounding
    double u = random_float(state);
    for (int64_t i = 0; i < n - 1; i++) {
        u -= p[i];
        if (u < 0.0) {
            return i;
        }
    }
    return n - 1;
}
