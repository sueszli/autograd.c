#include "../src/utils/async.h"
#include "../src/utils/types.h"
#include <stdatomic.h>
#include <stdbool.h>
#include <unistd.h>
#include <unity.h>

static atomic_int test_counter = 0;
static atomic_bool flags[10] = {false};

void setUp(void) {
    atomic_store(&test_counter, 0);
    for (int i = 0; i < 10; i++) {
        atomic_store(&flags[i], false);
    }
    async_cleanup_all();
}

void tearDown(void) {
    async_cleanup_all();
}

void simple_task(void) {
    atomic_fetch_add(&test_counter, 1);
}

void yield_task(void) {
    atomic_store(&flags[0], true);
    async_yield();
    atomic_store(&flags[1], true);
}

void interaction_task_1(void) {
    while (!atomic_load(&flags[0])) {
        async_yield();
    }
    atomic_store(&flags[1], true);
}

void interaction_task_2(void) {
    atomic_store(&flags[0], true);
}

void recursive_task(int depth) {
    if (depth > 0) {
        recursive_task(depth - 1);
    }
    if (depth == 0) {
        atomic_fetch_add(&test_counter, 1);
    }
}

void deep_recursion_task(void) {
    recursive_task(1000);
}

void test_async_spawn_single_thread(void) {
    async_spawn(simple_task);
    async_run_all();
    TEST_ASSERT_EQUAL(1, atomic_load(&test_counter));
}

void test_async_spawn_multiple_threads(void) {
    const int num_threads = 5;
    for (int i = 0; i < num_threads; i++) {
        async_spawn(simple_task);
    }
    async_run_all();
    TEST_ASSERT_EQUAL(num_threads, atomic_load(&test_counter));
}

void test_async_yield(void) {
    async_spawn(yield_task);
    async_run_all();
    TEST_ASSERT_TRUE(atomic_load(&flags[0]));
    TEST_ASSERT_TRUE(atomic_load(&flags[1]));
}

void test_async_thread_interaction(void) {
    async_spawn(interaction_task_1);
    async_spawn(interaction_task_2);
    async_run_all();
    TEST_ASSERT_TRUE(atomic_load(&flags[0]));
    TEST_ASSERT_TRUE(atomic_load(&flags[1]));
}

void test_async_deep_recursion(void) {
    async_spawn(deep_recursion_task);
    async_run_all();
    TEST_ASSERT_EQUAL(1, atomic_load(&test_counter));
}

void test_async_thread_termination(void) {
    u8 thread_id = async_spawn(simple_task);
    async_terminate_thread(thread_id);
    async_run_all();
    TEST_ASSERT_EQUAL(0, atomic_load(&test_counter));
}

void test_async_cleanup(void) {
    async_spawn(simple_task);
    async_cleanup_all();
    async_run_all();
    TEST_ASSERT_EQUAL(0, atomic_load(&test_counter));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_async_spawn_single_thread);
    RUN_TEST(test_async_spawn_multiple_threads);
    RUN_TEST(test_async_yield);
    RUN_TEST(test_async_thread_interaction);
    RUN_TEST(test_async_deep_recursion);
    RUN_TEST(test_async_thread_termination);
    RUN_TEST(test_async_cleanup);
    return UNITY_END();
}
