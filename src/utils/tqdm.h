#pragma once

#include "types.h"

#define TQDM_BAR_WIDTH 60

void tqdm(u64 current, u64 total, const char *desc, const char *info);
