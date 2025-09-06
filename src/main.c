#include <stdatomic.h>
#include <stdio.h>
#include <time.h>

#include "go.h"

#define NUM_TASKS 4

static void do_work(int task_id) {
    int result = 0;
    for (int i = 0; i < 1000000000; i++) {
        result += i % 7;
    }
    printf("Task %d done\n", task_id, result);
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main(void) {
    double start = get_time_ms();
    for (int i = 1; i <= NUM_TASKS; i++) {
        do_work(i);
    }
    double sync_time = get_time_ms() - start;
    printf("Synchronous total time: %.0f ms\n\n", sync_time);

    start = get_time_ms();
    for (int i = 1; i <= NUM_TASKS; i++) {
        go({ do_work(i); });
    }
    wait();
    double concurrent_time = get_time_ms() - start;
    printf("Concurrent total time: %.0f ms\n\n", concurrent_time);

    printf("Speedup: %.1fx faster with concurrent approach\n", sync_time / concurrent_time);

    return 0;
}
