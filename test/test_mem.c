#include "std/mem.h"
#include "unity.h"
#include <stdint.h>

void setUp(void) {}
void tearDown(void) {}

static unsigned char buf[64];

void test_alloc_is_zeroed_aligned_and_consecutive(void) {
    Arena a = {.buf = buf, .cap = sizeof(buf)};
    unsigned char *x = arena_alloc(&a, 3);
    x[0] = 1;
    unsigned char *y = arena_alloc(&a, 8);
    TEST_ASSERT_EQUAL(0, (uintptr_t)x % alignof(max_align_t));
    TEST_ASSERT_EQUAL(0, (uintptr_t)y % alignof(max_align_t));
    TEST_ASSERT_TRUE(y > x && y - x <= (ptrdiff_t)alignof(max_align_t));
    TEST_ASSERT_EQUAL(0, y[0]);
}

void test_reset_reuses_the_buffer_and_zeroes_again(void) {
    Arena a = {.buf = buf, .cap = sizeof(buf)};
    unsigned char *x = arena_alloc(&a, 8);
    x[0] = 1;
    arena_reset(&a);
    TEST_ASSERT_EQUAL(0, a.used);
    TEST_ASSERT_EQUAL_PTR(x, arena_alloc(&a, 8));
    TEST_ASSERT_EQUAL(0, x[0]);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_alloc_is_zeroed_aligned_and_consecutive);
    RUN_TEST(test_reset_reuses_the_buffer_and_zeroes_again);
    return UNITY_END();
}
