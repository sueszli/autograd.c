#pragma once

#include "cJSON.h"
#include <stdbool.h>
#include <stdint.h>

//
// json reading on top of cJSON, without malloc.
// the file and its parse tree live in static buffers.
// https://github.com/DaveGamble/cJSON
//
// example:
// --------
//     const cJSON *j = json_load("parity.json");
//     json_matrix(json_get(j, "wte"), wte, 27, 16, false);
//

// parses a whole file, aborts if it can't
const cJSON *json_load(const char *path);

// the field at key, aborts if it is missing
const cJSON *json_get(const cJSON *j, const char *key);

// reads a 2-D json array into out, a rows x cols row-major matrix
// if transposed, the json holds the cols x rows transpose instead
void json_matrix(const cJSON *j, double *out, int64_t rows, int64_t cols, bool transposed);
