#pragma once
#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

#define I8_MIN INT8_MIN
#define I8_MAX INT8_MAX
#define U8_MAX UINT8_MAX
#define I16_MIN INT16_MIN
#define I16_MAX INT16_MAX
#define U16_MAX UINT16_MAX
#define I32_MIN INT32_MIN
#define I32_MAX INT32_MAX
#define U32_MAX UINT32_MAX
#define I64_MIN INT64_MIN
#define I64_MAX INT64_MAX
#define U64_MAX UINT64_MAX

typedef float f32;
typedef double f64;

typedef uintptr_t uptr;
typedef intptr_t iptr;

typedef size_t usize;
typedef ptrdiff_t isize;

typedef uint_fast8_t u8_fast;
typedef uint_fast16_t u16_fast;
typedef uint_fast32_t u32_fast;
typedef uint_fast64_t u64_fast;

typedef int_fast8_t i8_fast;
typedef int_fast16_t i16_fast;
typedef int_fast32_t i32_fast;
typedef int_fast64_t i64_fast;

typedef char byte;
typedef uint8_t ubyte;

typedef void (*fn_ptr)(void);
typedef void *ptr;
