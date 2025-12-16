#pragma once

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#define TQDM_BAR_WIDTH 70
#define TQDM_PLOT_HEIGHT 15
#define TQDM_PLOT_WIDTH 78

// macro overloading to make the `prefix` and `postfix` args optional (default to `NULL`)
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

    printf("\r");
    if (prefix && prefix[0] != '\0') {
        printf("%s: ", prefix);
    }
    printf("%3u%%|", percentage);
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

// macro overloading for tqdm_plot with optional prefix
#define tqdm_plot(...) TQDM_PLOT_SELECT(__VA_ARGS__, tqdm_plot_4, tqdm_plot_3)(__VA_ARGS__)
#define TQDM_PLOT_SELECT(_1, _2, _3, _4, NAME, ...) NAME
#define tqdm_plot_3(current, total, loss) tqdm_plot_impl(current, total, loss, NULL)
#define tqdm_plot_4(current, total, loss, prefix) tqdm_plot_impl(current, total, loss, prefix)

static double plot_loss_history[TQDM_PLOT_WIDTH];
static int plot_history_count = 0;
static int plot_initialized = 0;

static inline void tqdm_plot_impl(uint64_t current, uint64_t total, double loss, const char *prefix) {
    if (total == 0)
        return;

    // plot spacing
    if (!plot_initialized) {
        for (int i = 0; i < TQDM_PLOT_HEIGHT + 2; i++) {
            printf("\n");
        }
        for (int i = 0; i < TQDM_PLOT_WIDTH; i++) {
            plot_loss_history[i] = 0.0;
        }
        plot_initialized = 1;
    }

    // update history buffer
    if (plot_history_count < TQDM_PLOT_WIDTH) {
        plot_loss_history[plot_history_count++] = loss;
    } else {
        for (int i = 0; i < TQDM_PLOT_WIDTH - 1; i++) {
            plot_loss_history[i] = plot_loss_history[i + 1];
        }
        plot_loss_history[TQDM_PLOT_WIDTH - 1] = loss;
    }

    // determine min/max for scaling
    double min_loss = plot_loss_history[0];
    double max_loss = plot_loss_history[0];
    for (int i = 1; i < plot_history_count; i++) {
        if (plot_loss_history[i] < min_loss)
            min_loss = plot_loss_history[i];
        if (plot_loss_history[i] > max_loss)
            max_loss = plot_loss_history[i];
    }

    if (max_loss == min_loss) {
        max_loss += 1e-6;
    }

    // move cursor up to redraw plot
    printf("\033[%dA", TQDM_PLOT_HEIGHT + 2);

    // frame top
    printf("\r\033[K");
    printf("┌");
    if (prefix && prefix[0] != '\0') {
        int title_len = 0;
        while (prefix[title_len])
            title_len++;
        int padding = (TQDM_PLOT_WIDTH - title_len - 2) / 2; // -2 for spaces around title
        if (padding < 0)
            padding = 0;

        for (int i = 0; i < padding; i++)
            printf("─");
        printf(" %s ", prefix);
        for (int i = 0; i < TQDM_PLOT_WIDTH - padding - title_len - 2; i++)
            printf("─");
    } else {
        for (int i = 0; i < TQDM_PLOT_WIDTH; i++)
            printf("─");
    }
    printf("┐\n");

    // render plot
    for (int row = TQDM_PLOT_HEIGHT - 1; row >= 0; row--) {
        printf("\r\033[K");
        printf("│");

        for (int col = 0; col < TQDM_PLOT_WIDTH; col++) {
            if (col >= plot_history_count) {
                printf(" ");
                continue;
            }

            double val = plot_loss_history[col];
            double normalized = (val - min_loss) / (max_loss - min_loss);
            int tick = (int)(normalized * (TQDM_PLOT_HEIGHT - 1));

            if (tick == row) {
                printf("•");
            } else if (row < tick) {
                printf(" "); // below the point
            } else {
                printf(" "); // above the point
            }
        }

        printf("│");

        // axis labels on the right side
        if (row == TQDM_PLOT_HEIGHT - 1)
            printf(" %.4f (max)", max_loss);
        if (row == 0)
            printf(" %.4f (min)", min_loss);

        printf("\n");
    }

    // frame bottom
    printf("\r\033[K");
    printf("└");
    for (int i = 0; i < TQDM_PLOT_WIDTH; i++)
        printf("─");
    printf("┘\n");

    tqdm(current, total, NULL, NULL);

    if (current >= total) {
        plot_initialized = 0;
        plot_history_count = 0;
    }
}
