#pragma once

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>

#define CONCAT(a, b) a##b
#define CONCAT_EXPAND(a, b) CONCAT(a, b)
#define UNIQUE_NAME(base) CONCAT_EXPAND(base, __LINE__)

typedef void (*goroutine_func_t)(void);

typedef struct {
    pthread_t thread;
    goroutine_func_t func;
    atomic_bool finished;
} goroutine_t;

static goroutine_t *goroutines[1024];
static atomic_int goroutine_count = 0;

static void *goroutine_wrapper(void *arg) {
    goroutine_t *g = (goroutine_t *)arg;
    g->func();
    atomic_store(&g->finished, true);
    return NULL;
}

static void spawn_goroutine(goroutine_func_t func) {
    goroutine_t *g = malloc(sizeof(goroutine_t));
    if (!g) {
        return;
    }

    g->func = func;
    atomic_store(&g->finished, false);

    int idx = atomic_fetch_add(&goroutine_count, 1);
    goroutines[idx] = g;

    pthread_create(&g->thread, NULL, goroutine_wrapper, g);
}

static void wait_for_goroutines(void) {
    int count = atomic_load(&goroutine_count);
    for (int i = 0; i < count; i++) {
        if (goroutines[i]) {
            pthread_join(goroutines[i]->thread, NULL);
            free(goroutines[i]);
            goroutines[i] = NULL;
        }
    }
    atomic_store(&goroutine_count, 0);
}

// clang-format off
#define go(block) \
    do { \
        void UNIQUE_NAME(__goroutine_func)(void) block \
        spawn_goroutine(UNIQUE_NAME(__goroutine_func)); \
    } while(0)
// clang-format on

#define wait() wait_for_goroutines()