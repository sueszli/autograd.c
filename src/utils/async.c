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

#define STACK_SIZE (2 << 15) // 64KB stack size

struct async_thread {
    ucontext_t context;         // cpu register state and stack pointer
    char *stack;                // dedicated stack memory
    fn_ptr func;                // function to execute
    async_thread_state_t state; // current execution state
    u8 id;                      // thread identifier
};

typedef struct async_thread uthread_t;

static uthread_t *threads[UINT8_MAX + 1] = {0}; // global thread pool
static u8 thread_count = 0;                     // total active threads
static u8 current_thread = 0;                   // currently executing thread ID
static ucontext_t main_context;                 // main program's execution context

static char *allocate_stack(size_t size) {
    // alloc virtual mem with all permissions
    void *stack = mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(stack != MAP_FAILED);
    // zero-init
    memset(stack, 0, size);
    return (char *)stack;
}

static void free_stack(char *stack, size_t size) {
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
    swapcontext(&threads[current_thread]->context, &main_context); // switch back to scheduler
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

    memset(&t->context, 0, sizeof(ucontext_t));      // zero-init context
    assert(getcontext(&t->context) != -1);           // capture current CPU state
    t->context.uc_stack.ss_sp = t->stack + 1024;     // start 1KB into allocation
    t->context.uc_stack.ss_size = STACK_SIZE - 2048; // leave 1KB at each end
    t->context.uc_link = NULL;                       // no return context
    makecontext(&t->context, invoke, 0);             // set entry point to wrapper

    threads[thread_count++] = t; // add to global thread pool
    return t->id;
}

void async_run_all(void) {
    // nothing to wait for
    if (thread_count == 0) {
        return;
    }

    // round robin scheduling
    while (true) {
        bool all_finished = true;

        for (u8 i = 0; i < thread_count; i++) {
            // null slot
            if (!threads[i]) {
                continue;
            }

            // shouldn't yield control if still running
            assert(threads[i]->state != ASYNC_THREAD_RUNNING);

            if (threads[i]->state == ASYNC_THREAD_READY || threads[i]->state == ASYNC_THREAD_YIELDED) {
                all_finished = false;
                current_thread = i;
                threads[i]->state = ASYNC_THREAD_RUNNING;

                // switch to thread context
                if (swapcontext(&main_context, &threads[i]->context) == -1) {
                    threads[i]->state = ASYNC_THREAD_FINISHED;
                    assert(false && "context switch failed");
                }
            }
        }

        if (all_finished) {
            break;
        }
    }

    // cleanup
    async_cleanup_all();
}

void async_terminate_thread(u8 thread_id) {
    assert(thread_id < thread_count && threads[thread_id] && "invalid thread ID");

    uthread_t *t = threads[thread_id];
    if (t->state != ASYNC_THREAD_FINISHED) {
        t->state = ASYNC_THREAD_FINISHED;
        free_stack(t->stack, STACK_SIZE);
        free(t);
        threads[thread_id] = NULL;
    }
}

void async_cleanup_all(void) {
    for (u8 i = 0; i < thread_count; i++) {
        if (threads[i]) {
            free_stack(threads[i]->stack, STACK_SIZE);
            free(threads[i]);
            threads[i] = NULL;
        }
    }
    thread_count = 0;
    current_thread = 0;
}
