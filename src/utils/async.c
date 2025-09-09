#define _GNU_SOURCE
#include "async.h"
#include "go.h"
#include "types.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <ucontext.h>
#include <unistd.h>

#define STACK_SIZE (2 << 15) // 64KB stack per thread

struct async_thread {
    ucontext_t context; // cpu register, stack pointer
    u8 *stack;          // stack memory
    fn_ptr func;        // function to execute
    async_thread_state_t state;
    u8 id;
};

typedef struct async_thread uthread_t;

static uthread_t *threads[UINT8_MAX + 1] = {0};
static u8 thread_count = 0;
static u8 current_thread = 0;
static ucontext_t main_context;

static u8 *allocate_stack(u64 size) {
    // with execute permissions, ASan doesn't complain about stack use when context switching
    void *stack = mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(stack != MAP_FAILED);
    memset(stack, 0, size);
    return (u8 *)stack;
}

static void free_stack(u8 *stack, u64 size) {
    if (!stack) {
        return;
    }
    munmap(stack, size);
}

void async_yield(void) {
    bool is_running = threads[current_thread] && threads[current_thread]->state == ASYNC_THREAD_RUNNING;
    if (!is_running) {
        return;
    }
    threads[current_thread]->state = ASYNC_THREAD_YIELDED;
    swapcontext(&threads[current_thread]->context, &main_context);
}

static void invoke(void) {
    uthread_t *t = threads[current_thread];
    assert(t != NULL);
    assert(t->func != NULL);
    t->func(); // exec
    t->state = ASYNC_THREAD_FINISHED;
    swapcontext(&t->context, &main_context);
}

u8 async_spawn(fn_ptr func) {
    assert(func);
    assert(thread_count < U8_MAX);

    uthread_t *t = malloc(sizeof(uthread_t));
    assert(t);

    t->stack = allocate_stack(STACK_SIZE);
    t->func = func;
    t->state = ASYNC_THREAD_READY;
    t->id = thread_count;
    assert(t->stack);

    memset(&t->context, 0, sizeof(ucontext_t));
    assert(getcontext(&t->context) != -1);
    // 1KB guard pages at each end of the stack to catch overflows
    t->context.uc_stack.ss_sp = t->stack + 1024;
    t->context.uc_stack.ss_size = STACK_SIZE - 2048;
    // no return context
    t->context.uc_link = NULL;
    makecontext(&t->context, invoke, 0);

    threads[thread_count++] = t;
    return t->id;
}

void async_run_all(void) {
    if (thread_count == 0) {
        return;
    }

    // round robin scheduling
    while (true) {
        bool all_finished = true;

        for (u8 i = 0; i < thread_count; i++) {
            // skip null slot
            if (!threads[i]) {
                continue;
            }

            // shouldn't yield control if still running
            assert(threads[i]->state != ASYNC_THREAD_RUNNING);

            // save this context, switch to thread's context
            if (threads[i]->state == ASYNC_THREAD_READY || threads[i]->state == ASYNC_THREAD_YIELDED) {
                all_finished = false;
                current_thread = i;
                threads[i]->state = ASYNC_THREAD_RUNNING;
                assert(swapcontext(&main_context, &threads[i]->context) != -1);
            }
        }

        if (all_finished) {
            break;
        }
    }

    async_cleanup_all();
}

void async_cleanup_all(void) {
    for (u8 i = 0; i < thread_count; i++) {
        if (threads[i]) {
            if (threads[i]->stack) {
                free_stack(threads[i]->stack, STACK_SIZE);
            }
            free(threads[i]);
            threads[i] = NULL;
        }
    }
    thread_count = 0;
    current_thread = 0;
}
