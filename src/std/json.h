#pragma once

#include "cJSON.h"
#include "mem.h"
#include "types.h"

//
// json reading on top of cJSON, into an arena instead of malloc.
// https://github.com/DaveGamble/cJSON
//

typedef struct {
    const cJSON *(*load)(const char *path, Arena *tree);
    const cJSON *(*get)(const cJSON *j, const char *key);

    // a 2-d json array into a rows x cols matrix, or from its transpose
    void (*matrix)(const cJSON *j, f64 *out, i64 rows, i64 cols, bool transposed);
} JsonNamespace;

extern const JsonNamespace json;
