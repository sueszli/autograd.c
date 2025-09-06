#include "go.h"
#include "types.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct {
    pthread_t thread;
    fn_ptr func;
    atomic_bool finished;
} goroutine_t;

static goroutine_t *go_routines[1024];
static _Atomic u16 goroutine_count = 0;

static void *wrap(void *arg) {
    goroutine_t *g = (goroutine_t *)arg;
    g->func();
    atomic_store(&g->finished, true);
    return NULL;
}

void spawn(fn_ptr func) {
    goroutine_t *g = malloc(sizeof(goroutine_t));
    if (!g) {
        return;
    }

    g->func = func;
    atomic_store(&g->finished, false);

    u16 idx = atomic_fetch_add(&goroutine_count, 1);
    go_routines[idx] = g;

    pthread_create(&g->thread, NULL, wrap, g);
}

void wait(void) {
    u16 count = atomic_load(&goroutine_count);
    for (u16 i = 0; i < count; i++) {
        if (go_routines[i]) {
            pthread_join(go_routines[i]->thread, NULL);
            free(go_routines[i]);
            go_routines[i] = NULL;
        }
    }
    atomic_store(&goroutine_count, 0);
}
