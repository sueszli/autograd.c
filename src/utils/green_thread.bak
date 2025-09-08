/*
 * green threads (user-space cooperative threads)
 *
 * only useful if you explicitly `yield()` control
 *
 * execution model:
 *
 * - `spawn()` creates thread structures but doesn't execute them
 * - threads remain in `THREAD_READY` state until `wait()` is called
 * - `wait()` acts as the scheduler, using round-robin to execute threads
 * - threads run cooperatively and must call `yield()` to give up CPU
 *
 * ```
 * spawn(func1);
 * spawn(func2);
 * wait(); // starts executing func1 and func2 concurrently
 * ```
 */

#define _GNU_SOURCE
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

typedef enum { THREAD_READY, THREAD_RUNNING, THREAD_FINISHED } thread_state_t;

typedef struct {
    ucontext_t context;   // cpu register state and stack pointer
    char *stack;          // dedicated stack memory
    fn_ptr func;          // function to execute
    thread_state_t state; // current execution state
    u8 id;                // thread identifier
} uthread_t;

static uthread_t *threads[UINT8_MAX + 1] = {0}; // global thread pool
static u8 thread_count = 0;                     // total active threads
static u8 current_thread = 0;                   // currently executing thread ID
static ucontext_t main_context;                 // main program's execution context

static char *allocate_stack(size_t size) {
    // allocate virtual memory with read/write/exec permissions
    void *stack = mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(stack != MAP_FAILED);
    // zero-init stack memory
    memset(stack, 0, size);
    return (char *)stack;
}

static void free_stack(char *stack, size_t size) {
    if (stack) {
        munmap(stack, size);
    }
}

void yield(void) {
    bool is_running = threads[current_thread] && threads[current_thread]->state == THREAD_RUNNING;
    if (is_running) {
        threads[current_thread]->state = THREAD_READY;
        swapcontext(&threads[current_thread]->context, &main_context); // switch back to scheduler
    }
}

static void invoke(void) {
    uthread_t *t = threads[current_thread];
    assert(t != NULL);
    assert(t->func != NULL);
    t->func(); // exec
    t->state = THREAD_FINISHED;
    swapcontext(&t->context, &main_context);
}

void spawn(fn_ptr func) {
    assert(func != NULL);
    assert(thread_count < UINT8_MAX);

    uthread_t *t = malloc(sizeof(uthread_t));
    t->stack = allocate_stack(STACK_SIZE);
    t->func = func;
    t->state = THREAD_READY;
    t->id = thread_count;
    assert(t != NULL);
    assert(t->stack != NULL);

    memset(&t->context, 0, sizeof(ucontext_t)); // zero-init context

    if (getcontext(&t->context) == -1) { // capture current CPU state
        free_stack(t->stack, STACK_SIZE);
        free(t);
        return;
    }

    // configure stack with ASan buffer zones
    t->context.uc_stack.ss_sp = t->stack + 1024;     // start 1KB into allocation
    t->context.uc_stack.ss_size = STACK_SIZE - 2048; // leave 1KB at each end
    t->context.uc_link = NULL;                       // no return context

    makecontext(&t->context, invoke, 0); // set entry point to wrapper

    threads[thread_count++] = t; // add to global thread pool
}

void wait(void) {
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
            assert(threads[i]->state != THREAD_RUNNING);

            if (threads[i]->state == THREAD_READY) {
                all_finished = false;
                current_thread = i;
                threads[i]->state = THREAD_RUNNING;

                // switch to thread context
                // returns -1 if thread finished execution
                if (swapcontext(&main_context, &threads[i]->context) == -1) {
                    threads[i]->state = THREAD_FINISHED;
                }
            }
        }

        if (all_finished) {
            break;
        }
    }

    // cleanup
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
