#pragma once
#include "types.h"

typedef struct async_thread async_thread_t;

typedef enum { ASYNC_THREAD_READY, ASYNC_THREAD_RUNNING, ASYNC_THREAD_FINISHED, ASYNC_THREAD_YIELDED } async_thread_state_t;

u8 async_spawn(fn_ptr func);

void async_yield(void);

// event-loop like async using ucontext.h
void async_run_all(void);

void async_cleanup_all(void);
