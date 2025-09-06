#pragma once

#include "types.h"

#define CONCAT(a, b) a##b
#define CONCAT_EXPAND(a, b) CONCAT(a, b)
#define UNIQUE_NAME(base) CONCAT_EXPAND(base, __LINE__)

// clang-format off
#define defer(block) \
    void UNIQUE_NAME(__cleanup_)(u8 *ptr __attribute__((unused))) { block } \
    u8 UNIQUE_NAME(__defer_var_) __attribute__((cleanup(UNIQUE_NAME(__cleanup_)))) = 0
// clang-format on
