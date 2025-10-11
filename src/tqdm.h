#pragma once

#include <inttypes.h>
#include <stdio.h>
#include <sys/time.h>

#define TQDM_BAR_WIDTH 60

// macro overloading magic to make the prefix and postfix optional (default `to` NULL)
#define tqdm(...) TQDM_SELECT(__VA_ARGS__, tqdm_4, tqdm_3, tqdm_2)(__VA_ARGS__)
#define TQDM_SELECT(_1, _2, _3, _4, NAME, ...) NAME // selects macro based on arg count
#define tqdm_2(current, total) tqdm_impl(current, total, NULL, NULL)
#define tqdm_3(current, total, prefix) tqdm_impl(current, total, prefix, NULL)
#define tqdm_4(current, total, prefix, postfix) tqdm_impl(current, total, prefix, postfix)

static struct timeval start_time = {0, 0};

static inline void tqdm_impl(uint64_t current, uint64_t total, const char *prefix, const char *postfix) {
    if (total == 0) {
        return;
    }

    if (start_time.tv_sec == 0 && start_time.tv_usec == 0) {
        gettimeofday(&start_time, NULL);
    }

    const double progress = (double)current / (double)total;
    const uint32_t percentage = (uint32_t)(progress * 100.0);
    const uint32_t bar_width = TQDM_BAR_WIDTH;
    const uint32_t filled = (uint32_t)(progress * bar_width);

    struct timeval now;
    gettimeofday(&now, NULL);
    const double elapsed = (double)(now.tv_sec - start_time.tv_sec) + (double)(now.tv_usec - start_time.tv_usec) / 1e6;
    const double rate = (elapsed > 0) ? (double)current / elapsed : 0.0;

    printf("\r%s: %3u%%|", prefix ? prefix : "Progress", percentage);
    for (uint32_t i = 0; i < filled; i++) {
        printf("█");
    }
    if (filled < bar_width) {
        double partial = (progress * bar_width) - filled;
        if (partial > 0.75) {
            printf("▊");
        } else if (partial > 0.5) {
            printf("▌");
        } else if (partial > 0.25) {
            printf("▎");
        } else {
            printf("▏");
        }
        for (uint32_t i = filled + 1; i < bar_width; i++) {
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
