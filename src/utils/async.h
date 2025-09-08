#pragma once
#include "types.h"

// cooperative threading library for asynchronous execution
typedef struct async_thread async_thread_t;

typedef enum { ASYNC_THREAD_READY, ASYNC_THREAD_RUNNING, ASYNC_THREAD_FINISHED, ASYNC_THREAD_YIELDED } async_thread_state_t;

// creates a new thread
u8 async_spawn(fn_ptr func);

// yields control to the scheduler
void async_yield(void);

// runs all spawned threads until completion
void async_run_all(void);

// terminates a specific thread by ID
void async_terminate_thread(u8 thread_id);

// terminates all threads and frees resources
void async_cleanup_all(void);
