#pragma once

#include "cJSON.h"
#include "mem.h"
#include "types.h"

//
// json reading on top of cJSON, without malloc.
// the file and its parse tree live in an arena the caller owns.
// https://github.com/DaveGamble/cJSON
//
// example:
// --------
//     const cJSON *j = json.load("parity.json", &tree);
//     json.matrix(json.get(j, "wte"), wte, 27, 16, false);
//

typedef struct {
    // parses a whole file into tree, aborts if it can't
    const cJSON *(*load)(const char *path, Arena *tree);

    // the field at key, aborts if it is missing
    const cJSON *(*get)(const cJSON *j, const char *key);

    // reads a 2-D json array into out, a rows x cols row-major matrix
    // if transposed, the json holds the cols x rows transpose instead
    void (*matrix)(const cJSON *j, f64 *out, i64 rows, i64 cols, bool transposed);
} JsonNamespace;

extern const JsonNamespace json;
