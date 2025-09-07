#include "../src/utils/go.h"
#include "../src/utils/types.h"
#include <float.h>
#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>
#include <unity.h>

static atomic_int test_counter = 0;
static atomic_bool test_flag = false;
static atomic_int execution_order[10];
static atomic_int execution_index = 0;

void setUp(void) {
    atomic_store(&test_counter, 0);
    atomic_store(&test_flag, false);
    atomic_store(&execution_index, 0);
    for (i32 i = 0; i < 10; i++) {
        atomic_store(&execution_order[i], -1);
    }
}

void tearDown(void) { wait(); }

void test_go_single_goroutine(void) {
    go({
        atomic_store(&test_flag, true);
        atomic_fetch_add(&test_counter, 1);
    });

    wait();

    TEST_ASSERT_TRUE(atomic_load(&test_flag));
    TEST_ASSERT_EQUAL(1, atomic_load(&test_counter));
}

void test_go_multiple_goroutines(void) {
    const i32 num_goroutines = 5;

    for (i32 i = 0; i < num_goroutines; i++) {
        go({ atomic_fetch_add(&test_counter, 1); });
    }

    wait();

    TEST_ASSERT_EQUAL(num_goroutines, atomic_load(&test_counter));
}

void test_go_concurrent_execution(void) {
    const u32 delay_ms = 50;

    go({
        usleep(delay_ms * 1000);
        i32 idx = atomic_fetch_add(&execution_index, 1);
        atomic_store(&execution_order[idx], 1);
    });

    go({
        usleep(delay_ms * 1000);
        i32 idx = atomic_fetch_add(&execution_index, 1);
        atomic_store(&execution_order[idx], 2);
    });

    go({
        usleep(delay_ms * 1000);
        i32 idx = atomic_fetch_add(&execution_index, 1);
        atomic_store(&execution_order[idx], 3);
    });

    wait();

    TEST_ASSERT_EQUAL(3, atomic_load(&execution_index));

    i32 executed_goroutines[3] = {atomic_load(&execution_order[0]), atomic_load(&execution_order[1]), atomic_load(&execution_order[2])};

    bool found_1 = false, found_2 = false, found_3 = false;
    for (i32 i = 0; i < 3; i++) {
        if (executed_goroutines[i] == 1)
            found_1 = true;
        if (executed_goroutines[i] == 2)
            found_2 = true;
        if (executed_goroutines[i] == 3)
            found_3 = true;
    }

    TEST_ASSERT_TRUE(found_1);
    TEST_ASSERT_TRUE(found_2);
    TEST_ASSERT_TRUE(found_3);
}

void test_go_nested_goroutines(void) {
    go({
        atomic_fetch_add(&test_counter, 1);

        go({ atomic_fetch_add(&test_counter, 10); });
    });

    wait();

    TEST_ASSERT_EQUAL(11, atomic_load(&test_counter));
}

void test_go_empty_block(void) {
    go({
        // nothing to do here
    });

    wait();

    TEST_ASSERT_TRUE(true);
}

void test_go_variable_capture(void) {
    atomic_int captured_value = 0;
    i32 local_value = 42;

    go({ atomic_store(&captured_value, local_value); });

    wait();

    TEST_ASSERT_EQUAL(42, atomic_load(&captured_value));
}

void test_go_race_condition_safety(void) {
    const i32 num_increments = 100;

    for (i32 i = 0; i < num_increments; i++) {
        go({ atomic_fetch_add(&test_counter, 1); });
    }

    wait();

    TEST_ASSERT_EQUAL(num_increments, atomic_load(&test_counter));
}

void test_go_sequential_wait_calls(void) {
    go({ atomic_fetch_add(&test_counter, 1); });

    wait();
    TEST_ASSERT_EQUAL(1, atomic_load(&test_counter));

    go({ atomic_fetch_add(&test_counter, 2); });

    wait();
    TEST_ASSERT_EQUAL(3, atomic_load(&test_counter));

    go({ atomic_fetch_add(&test_counter, 3); });

    wait();
    TEST_ASSERT_EQUAL(6, atomic_load(&test_counter));
}

void test_go_goroutine_isolation(void) {
    atomic_int counter1 = 0;
    atomic_int counter2 = 0;

    go({
        for (i32 i = 0; i < 10; i++) {
            atomic_fetch_add(&counter1, 1);
            usleep(1000);
        }
    });

    go({
        for (i32 i = 0; i < 15; i++) {
            atomic_fetch_add(&counter2, 1);
            usleep(1000);
        }
    });

    wait();

    TEST_ASSERT_EQUAL(10, atomic_load(&counter1));
    TEST_ASSERT_EQUAL(15, atomic_load(&counter2));
}

void test_go_large_number_of_goroutines(void) {
    const i32 num_goroutines = 50;

    for (i32 i = 0; i < num_goroutines; i++) {
        go({
            atomic_fetch_add(&test_counter, 1);
            usleep(1000);
        });
    }

    wait();

    TEST_ASSERT_EQUAL(num_goroutines, atomic_load(&test_counter));
}

i32 main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_go_single_goroutine);
    RUN_TEST(test_go_multiple_goroutines);
    RUN_TEST(test_go_concurrent_execution);
    RUN_TEST(test_go_nested_goroutines);
    RUN_TEST(test_go_empty_block);
    RUN_TEST(test_go_variable_capture);
    RUN_TEST(test_go_race_condition_safety);
    RUN_TEST(test_go_sequential_wait_calls);
    RUN_TEST(test_go_goroutine_isolation);
    RUN_TEST(test_go_large_number_of_goroutines);

    return UNITY_END();
}