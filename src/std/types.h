#pragma once

#include <float.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

//
// rust-style fixed-width types.
// https://doc.rust-lang.org/book/ch03-02-data-types.html
//
// the static asserts check every width at compile time.
// on a system where one differs, the build fails before anything runs.
//

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef _Float16 f16;
typedef float f32;
typedef double f64;

typedef ptrdiff_t isize;
typedef size_t usize;

// bool, true and false come from stdbool.h, and are keywords from c23 on

static_assert(CHAR_BIT == 8, "a byte must have 8 bits");

static_assert(sizeof(i8) == 1 && sizeof(u8) == 1, "i8 and u8 must be 8 bits");
static_assert(sizeof(i16) == 2 && sizeof(u16) == 2, "i16 and u16 must be 16 bits");
static_assert(sizeof(i32) == 4 && sizeof(u32) == 4, "i32 and u32 must be 32 bits");
static_assert(sizeof(i64) == 8 && sizeof(u64) == 8, "i64 and u64 must be 64 bits");

static_assert(sizeof(f16) == 2, "f16 must be 16 bits");
static_assert(sizeof(f32) == 4 && FLT_MANT_DIG == 24, "f32 must be ieee 754 binary32");
static_assert(sizeof(f64) == 8 && DBL_MANT_DIG == 53, "f64 must be ieee 754 binary64");

static_assert(sizeof(bool) == 1, "bool must be 1 byte");

static_assert(sizeof(isize) == sizeof(void *) && sizeof(usize) == sizeof(void *), "isize and usize must be pointer-sized");
