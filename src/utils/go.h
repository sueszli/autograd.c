#pragma once

#include "types.h"

#define CONCAT(a, b) a##b
#define CONCAT_EXPAND(a, b) CONCAT(a, b)
#define UNIQUE_NAME(base) CONCAT_EXPAND(base, __LINE__)

void spawn(fn_ptr func);

// clang-format off
#define go(block) \
    do { \
        void UNIQUE_NAME(func)(void) block \
        spawn(UNIQUE_NAME(func)); \
    } while(0)
// clang-format on

void wait(void);
