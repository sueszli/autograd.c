#pragma once

#define CONCAT(a, b) a##b
#define CONCAT_EXPAND(a, b) CONCAT(a, b)
#define UNIQUE_NAME(base) CONCAT_EXPAND(base, __LINE__)

// clang-format off
#define defer(block) \
    void UNIQUE_NAME(__cleanup_)(int *ptr __attribute__((unused))) { block } \
    int UNIQUE_NAME(__defer_var_) __attribute__((cleanup(UNIQUE_NAME(__cleanup_)))) = 0
// clang-format on
