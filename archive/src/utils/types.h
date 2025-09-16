#pragma once
#include <stddef.h>
#include <stdint.h>

//
// signed integer types
//

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef int_fast8_t i8_fast;
typedef int_fast16_t i16_fast;
typedef int_fast32_t i32_fast;
typedef int_fast64_t i64_fast;

#define I8_MIN INT8_MIN
#define I8_MAX INT8_MAX
#define I16_MIN INT16_MIN
#define I16_MAX INT16_MAX
#define I32_MIN INT32_MIN
#define I32_MAX INT32_MAX
#define I64_MIN INT64_MIN
#define I64_MAX INT64_MAX

//
// unsigned integer types
//

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef uint_fast8_t u8_fast;
typedef uint_fast16_t u16_fast;
typedef uint_fast32_t u32_fast;
typedef uint_fast64_t u64_fast;

#define U8_MAX UINT8_MAX
#define U16_MAX UINT16_MAX
#define U32_MAX UINT32_MAX
#define U64_MAX UINT64_MAX

//
// floating point types
//

typedef float f32;
typedef double f64;

#define F32_MAX FLT_MAX
#define F32_MIN FLT_MIN
#define F64_MAX DBL_MAX
#define F64_MIN DBL_MIN
#define F32_EPSILON FLT_EPSILON
#define F64_EPSILON DBL_EPSILON
#define F32_INFINITY INFINITY
#define F64_INFINITY INFINITY
#define F32_NAN NAN
#define F64_NAN NAN

//
// pointers
//

typedef void (*fn_ptr)(void);

//
// assert 64-bit architecture
//

_Static_assert(sizeof(i8) == 1);
_Static_assert(sizeof(i16) == 2);
_Static_assert(sizeof(i32) == 4);
_Static_assert(sizeof(i64) == 8);
_Static_assert(sizeof(u8) == 1);
_Static_assert(sizeof(u16) == 2);
_Static_assert(sizeof(u32) == 4);
_Static_assert(sizeof(u64) == 8);
_Static_assert(sizeof(f32) == 4);
_Static_assert(sizeof(f64) == 8);

_Static_assert(sizeof(char) == sizeof(u8));
_Static_assert(sizeof(signed char) == sizeof(i8));
_Static_assert(sizeof(unsigned char) == sizeof(u8));
_Static_assert(sizeof(short) == sizeof(i16));
_Static_assert(sizeof(unsigned short) == sizeof(u16));
_Static_assert(sizeof(int) == sizeof(i32));
_Static_assert(sizeof(unsigned int) == sizeof(u32));
_Static_assert(sizeof(long) == sizeof(i64));
_Static_assert(sizeof(unsigned long) == sizeof(u64));
_Static_assert(sizeof(long long) == sizeof(i64));
_Static_assert(sizeof(unsigned long long) == sizeof(u64));
_Static_assert(sizeof(float) == sizeof(f32));
_Static_assert(sizeof(double) == sizeof(f64));
_Static_assert(sizeof(size_t) == sizeof(u64));
_Static_assert(sizeof(ptrdiff_t) == sizeof(i64));
_Static_assert(sizeof(uintptr_t) == sizeof(u64));
_Static_assert(sizeof(intptr_t) == sizeof(i64));
_Static_assert(sizeof(void *) == sizeof(u64));
_Static_assert(sizeof(fn_ptr) == sizeof(u64));
