#include "go.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>

typedef void (*func_t)(void);

typedef struct {
    pthread_t thread;
    func_t func;
    atomic_bool finished;
} goroutine_t;

static goroutine_t *go_routines[1024];
static atomic_int go_routine_count = 0;

static void *wrap(void *arg) {
    goroutine_t *g = (goroutine_t *)arg;
    g->func();
    atomic_store(&g->finished, true);
    return NULL;
}

void spawn(func_t func) {
    goroutine_t *g = malloc(sizeof(goroutine_t));
    if (!g) {
        return;
    }

    g->func = func;
    atomic_store(&g->finished, false);

    int idx = atomic_fetch_add(&go_routine_count, 1);
    go_routines[idx] = g;

    pthread_create(&g->thread, NULL, wrap, g);
}

void wait(void) {
    int count = atomic_load(&go_routine_count);
    for (int i = 0; i < count; i++) {
        if (go_routines[i]) {
            pthread_join(go_routines[i]->thread, NULL);
            free(go_routines[i]);
            go_routines[i] = NULL;
        }
    }
    atomic_store(&go_routine_count, 0);
}
