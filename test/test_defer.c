#include "../src/utils/defer.h"
#include "../src/utils/types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unity.h>

static i32 cleanup_order = 0; // cleanup execution order
static i32 cleanup_results[10];

void setUp(void) {
    cleanup_order = 0;
    memset(cleanup_results, 0, sizeof(cleanup_results));
}

void tearDown(void) {}

void test_basic_defer(void) {
    i32 cleaned_up = 0;

    {
        defer({ cleaned_up = 1; });
        TEST_ASSERT_EQUAL(0, cleaned_up); // should not be cleaned up yet
    }

    TEST_ASSERT_EQUAL(1, cleaned_up); // should be cleaned up after scope exit
}

void test_multiple_defers(void) {
    {
        defer({ cleanup_results[cleanup_order++] = 1; });
        defer({ cleanup_results[cleanup_order++] = 2; });
        defer({ cleanup_results[cleanup_order++] = 3; });
    }

    // defers should execute in reverse order (LIFO)
    TEST_ASSERT_EQUAL(3, cleanup_order);
    TEST_ASSERT_EQUAL(3, cleanup_results[0]);
    TEST_ASSERT_EQUAL(2, cleanup_results[1]);
    TEST_ASSERT_EQUAL(1, cleanup_results[2]);
}

void test_defer_with_variables(void) {
    char *buffer = malloc(100);
    i32 freed = 0;

    {
        defer({
            free(buffer);
            freed = 1;
        });

        strcpy(buffer, "test string");
        TEST_ASSERT_EQUAL_STRING("test string", buffer);
        TEST_ASSERT_EQUAL(0, freed);
    }

    TEST_ASSERT_EQUAL(1, freed);
}

void test_defer_in_function(void) {
    void test_function() {
        defer({ cleanup_results[cleanup_order++] = 42; });
        TEST_ASSERT_EQUAL(0, cleanup_order);
    }

    test_function();
    TEST_ASSERT_EQUAL(1, cleanup_order);
    TEST_ASSERT_EQUAL(42, cleanup_results[0]);
}

void test_nested_scopes(void) {
    {
        defer({ cleanup_results[cleanup_order++] = 1; });

        {
            defer({ cleanup_results[cleanup_order++] = 2; });

            {
                defer({ cleanup_results[cleanup_order++] = 3; });
            }

            TEST_ASSERT_EQUAL(1, cleanup_order);
            TEST_ASSERT_EQUAL(3, cleanup_results[0]);
        }

        TEST_ASSERT_EQUAL(2, cleanup_order);
        TEST_ASSERT_EQUAL(2, cleanup_results[1]);
    }

    TEST_ASSERT_EQUAL(3, cleanup_order);
    TEST_ASSERT_EQUAL(1, cleanup_results[2]);
}

void test_defer_with_complex_block(void) {
    FILE *file = NULL;
    i32 *array = NULL;
    i32 success = 0;

    {
        defer({
            if (file) {
                fclose(file);
            }
            if (array) {
                free(array);
            }
            success = 1;
        });

        array = malloc(sizeof(i32) * 10);
        TEST_ASSERT_NOT_NULL(array);

        for (i32 i = 0; i < 10; i++) {
            array[i] = i * i;
        }

        TEST_ASSERT_EQUAL(0, success);
    }

    TEST_ASSERT_EQUAL(1, success);
}

void test_defer_unique_names(void) {
    i32 result1 = 0, result2 = 0;

    {
        defer({ result1 = 1; });
        defer({ result2 = 2; });
        TEST_ASSERT_EQUAL(0, result1);
        TEST_ASSERT_EQUAL(0, result2);
    }

    TEST_ASSERT_EQUAL(1, result1);
    TEST_ASSERT_EQUAL(2, result2);
}

i32 main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_basic_defer);
    RUN_TEST(test_multiple_defers);
    RUN_TEST(test_defer_with_variables);
    RUN_TEST(test_defer_in_function);
    RUN_TEST(test_nested_scopes);
    RUN_TEST(test_defer_with_complex_block);
    RUN_TEST(test_defer_unique_names);

    return UNITY_END();
}
