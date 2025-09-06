#include "go.h"
#include "types.h"

#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct {
    pthread_t thread;
    fn_ptr func;
    atomic_bool finished;
} goroutine_t;

static goroutine_t *goroutines[UINT8_MAX + 1] = {0};
static _Atomic u8 goroutine_count = 0; // smaller means faster wait()

static void *invoke(void *arg) {
    goroutine_t *g = (goroutine_t *)arg;
    assert(g != NULL);
    assert(g->func != NULL);
    g->func(); // call
    atomic_store(&g->finished, true);
    return NULL;
}

void spawn(fn_ptr func) {
    assert(func != NULL);

    u8 current_count = atomic_load(&goroutine_count);
    assert(current_count < UINT8_MAX);

    goroutine_t *g = malloc(sizeof(goroutine_t));
    assert(g != NULL);

    g->func = func;
    atomic_store(&g->finished, false);

    u8 idx = atomic_fetch_add(&goroutine_count, 1);
    assert(idx < UINT8_MAX);

    goroutines[idx] = g;

    int result = pthread_create(&g->thread, NULL, invoke, g);
    assert(result == 0);
}

void wait(void) {
    // barrier
    bool all_finished = false;
    while (!all_finished) {
        all_finished = true;
        u8 current_count = atomic_load(&goroutine_count);
        for (u8 i = 0; i < current_count; i++) {
            bool not_finished = goroutines[i] && !atomic_load(&goroutines[i]->finished);
            if (not_finished) {
                all_finished = false;
                sched_yield(); // avoid busy waiting
                break;
            }
        }
    }

    // cleanup
    u8 final_count = atomic_load(&goroutine_count);
    for (u8 i = 0; i < final_count; i++) {
        bool doesnt_exist = goroutines[i] == NULL;
        if (doesnt_exist) {
            continue;
        }
        int result = pthread_join(goroutines[i]->thread, NULL);
        assert(result == 0);
        free(goroutines[i]);
        goroutines[i] = NULL;
    }
    atomic_store(&goroutine_count, 0);
}
