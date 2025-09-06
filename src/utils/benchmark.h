#pragma once

#include <stdio.h>
#include <time.h>

// clang-format off
#define benchmark(name, block) \
    do { \
        struct timespec start, end; \
        clock_gettime(CLOCK_MONOTONIC, &start); \
        block; \
        clock_gettime(CLOCK_MONOTONIC, &end); \
        double time_spent = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9; \
        printf(name " took %.3f seconds to execute\n", time_spent); \
    } while (0)
// clang-format on
