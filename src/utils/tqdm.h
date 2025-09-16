#pragma once

#include "defer.h"
#include "types.h"

#include <inttypes.h>
#include <stdio.h>
#include <sys/time.h>

#define TQDM_BAR_WIDTH 60

static struct timeval start_time = {0, 0};

static inline void tqdm(u64 current, u64 total, const char *prefix, const char *postfix) {
    if (start_time.tv_sec == 0 && start_time.tv_usec == 0) {
        gettimeofday(&start_time, NULL);
    }

    f64 progress = (f64)current / (f64)total;
    u32 percentage = (u32)(progress * 100.0);
    u32 bar_width = TQDM_BAR_WIDTH;
    u32 filled = (u32)(progress * bar_width);

    struct timeval now;
    gettimeofday(&now, NULL);
    f64 elapsed = (f64)(now.tv_sec - start_time.tv_sec) + (f64)(now.tv_usec - start_time.tv_usec) / 1e6;
    f64 rate = (elapsed > 0) ? (f64)current / elapsed : 0.0;

    printf("\r%s: %3u%%|", prefix ? prefix : "Progress", percentage);
    for (u32 i = 0; i < filled; i++) {
        printf("█");
    }
    if (filled < bar_width) {
        f64 partial = (progress * bar_width) - filled;
        if (partial > 0.75) {
            printf("▊");
        } else if (partial > 0.5) {
            printf("▌");
        } else if (partial > 0.25) {
            printf("▎");
        } else {
            printf("▏");
        }
        for (u32 i = filled + 1; i < bar_width; i++) {
            printf(" ");
        }
    }
    printf("| %" PRIu64 "/%" PRIu64 " [%.1fit/s]", current, total, rate);
    if (postfix && postfix[0] != '\0') {
        printf(" %s", postfix);
    }
    printf("   ");
    fflush(stdout);

    // reset for next use
    if (current >= total) {
        start_time.tv_sec = 0;
        start_time.tv_usec = 0;
        printf("\n");
    }
}
