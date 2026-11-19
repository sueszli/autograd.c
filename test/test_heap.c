#include "std/heap.h"
#include "unity.h"
#include <stdint.h>

void setUp(void) {}
void tearDown(void) {}

static alignas(16) unsigned char buf[64];

void test_alloc_bumps_and_returns_null_when_full(void) {
    FixedBufferAllocator fba = fixed_buffer_allocator_init(buf, sizeof(buf));
    TEST_ASSERT_EQUAL_PTR(buf, fixed_buffer_allocator_alloc(&fba, 40, 1));
    TEST_ASSERT_EQUAL_PTR(buf + 40, fixed_buffer_allocator_alloc(&fba, 24, 1));
    TEST_ASSERT_NULL(fixed_buffer_allocator_alloc(&fba, 1, 1));
    TEST_ASSERT_EQUAL_size_t(64, fba.end_index);
}

void test_alloc_pads_to_alignment(void) {
    FixedBufferAllocator fba = fixed_buffer_allocator_init(buf, sizeof(buf));
    fixed_buffer_allocator_alloc(&fba, 3, 1);
    unsigned char *p = fixed_buffer_allocator_alloc(&fba, 8, 16);
    TEST_ASSERT_EQUAL_PTR(buf + 16, p);
    TEST_ASSERT_EQUAL_size_t(24, fba.end_index);
    // padding counts against capacity: 24 rounds up to 32, and 32 + 40 > 64
    TEST_ASSERT_NULL(fixed_buffer_allocator_alloc(&fba, 40, 16));
    TEST_ASSERT_EQUAL_size_t(24, fba.end_index); // padding alone runs past the end
    fixed_buffer_allocator_alloc(&fba, 35, 1);
    TEST_ASSERT_NULL(fixed_buffer_allocator_alloc(&fba, 1, 16));
}

void test_free_only_moves_end_index_for_the_last_allocation(void) {
    FixedBufferAllocator fba = fixed_buffer_allocator_init(buf, sizeof(buf));
    unsigned char *a = fixed_buffer_allocator_alloc(&fba, 8, 8);
    unsigned char *b = fixed_buffer_allocator_alloc(&fba, 8, 8);
    fixed_buffer_allocator_free(&fba, a, 8);
    TEST_ASSERT_EQUAL_size_t(16, fba.end_index);
    fixed_buffer_allocator_free(&fba, b, 8);
    TEST_ASSERT_EQUAL_size_t(8, fba.end_index);
}

void test_resize(void) {
    FixedBufferAllocator fba = fixed_buffer_allocator_init(buf, sizeof(buf));
    unsigned char *a = fixed_buffer_allocator_alloc(&fba, 8, 8);
    unsigned char *b = fixed_buffer_allocator_alloc(&fba, 8, 8);
    // not the last allocation: shrinking succeeds without freeing anything, growing fails
    TEST_ASSERT_TRUE(fixed_buffer_allocator_resize(&fba, a, 8, 4));
    TEST_ASSERT_FALSE(fixed_buffer_allocator_resize(&fba, a, 8, 9));
    TEST_ASSERT_EQUAL_size_t(16, fba.end_index);
    // the last allocation grows and shrinks in place, up to the end of the buffer
    TEST_ASSERT_TRUE(fixed_buffer_allocator_resize(&fba, b, 8, 48));
    TEST_ASSERT_EQUAL_size_t(56, fba.end_index);
    TEST_ASSERT_FALSE(fixed_buffer_allocator_resize(&fba, b, 48, 57));
    TEST_ASSERT_TRUE(fixed_buffer_allocator_resize(&fba, b, 48, 56));
    TEST_ASSERT_EQUAL_size_t(64, fba.end_index);
    TEST_ASSERT_TRUE(fixed_buffer_allocator_resize(&fba, b, 56, 2));
    TEST_ASSERT_EQUAL_size_t(10, fba.end_index);
}

void test_reset(void) {
    FixedBufferAllocator fba = fixed_buffer_allocator_init(buf, sizeof(buf));
    fixed_buffer_allocator_alloc(&fba, 64, 1);
    fixed_buffer_allocator_reset(&fba);
    TEST_ASSERT_EQUAL_PTR(buf, fixed_buffer_allocator_alloc(&fba, 64, 1));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_alloc_bumps_and_returns_null_when_full);
    RUN_TEST(test_alloc_pads_to_alignment);
    RUN_TEST(test_free_only_moves_end_index_for_the_last_allocation);
    RUN_TEST(test_resize);
    RUN_TEST(test_reset);
    return UNITY_END();
}
