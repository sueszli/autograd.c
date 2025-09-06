#pragma once

#include "types.h"
#include <stdint.h>
#include <stdio.h>
#include <time.h>

// clang-format off
#define benchmark(name, block) \
    do { \
        struct timespec start, end; \
        clock_gettime(CLOCK_MONOTONIC, &start); \
        block; \
        clock_gettime(CLOCK_MONOTONIC, &end); \
        f64 time_spent = (f64)(end.tv_sec - start.tv_sec) + (f64)(end.tv_nsec - start.tv_nsec) / 1e9; \
        printf(name " took %.3f seconds to execute\n", time_spent); \
    } while (0)
// clang-format on
