#include "../src/defer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cleanup_order = 0; // cleanup execution order
static int cleanup_results[10];

void reset_cleanup_tracking() {
    cleanup_order = 0;
    memset(cleanup_results, 0, sizeof(cleanup_results));
}

void test_basic_defer() {
    reset_cleanup_tracking();

    int cleaned_up = 0;

    {
        defer({ cleaned_up = 1; });

        assert(cleaned_up == 0); // should not be cleaned up yet
    }

    assert(cleaned_up == 1); // should be cleaned up after scope exit
}

void test_multiple_defers() {
    reset_cleanup_tracking();

    {
        defer({ cleanup_results[cleanup_order++] = 1; });

        defer({ cleanup_results[cleanup_order++] = 2; });

        defer({ cleanup_results[cleanup_order++] = 3; });
    }

    // defers should execute in reverse order (LIFO)
    assert(cleanup_order == 3);
    assert(cleanup_results[0] == 3);
    assert(cleanup_results[1] == 2);
    assert(cleanup_results[2] == 1);
}

void test_defer_with_variables() {
    char *buffer = malloc(100);
    int freed = 0;

    {
        defer({
            free(buffer);
            freed = 1;
        });

        strcpy(buffer, "test string");
        assert(strcmp(buffer, "test string") == 0);
        assert(freed == 0);
    }

    assert(freed == 1);
}

void test_defer_in_function() {
    reset_cleanup_tracking();

    void test_function() {
        defer({ cleanup_results[cleanup_order++] = 42; });

        assert(cleanup_order == 0);
    }

    test_function();
    assert(cleanup_order == 1);
    assert(cleanup_results[0] == 42);
}

void test_nested_scopes() {
    reset_cleanup_tracking();

    {
        defer({ cleanup_results[cleanup_order++] = 1; });

        {
            defer({ cleanup_results[cleanup_order++] = 2; });

            {
                defer({ cleanup_results[cleanup_order++] = 3; });
            }

            assert(cleanup_order == 1);
            assert(cleanup_results[0] == 3);
        }

        assert(cleanup_order == 2);
        assert(cleanup_results[1] == 2);
    }

    assert(cleanup_order == 3);
    assert(cleanup_results[2] == 1);
}

void test_defer_with_complex_block() {
    FILE *file = NULL;
    int *array = NULL;
    int success = 0;

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

        array = malloc(sizeof(int) * 10);
        assert(array != NULL);

        for (int i = 0; i < 10; i++) {
            array[i] = i * i;
        }

        assert(success == 0);
    }

    assert(success == 1);
}

void test_defer_unique_names() {
    int result1 = 0, result2 = 0;

    {
        defer({ result1 = 1; });
        defer({ result2 = 2; });

        assert(result1 == 0 && result2 == 0);
    }

    assert(result1 == 1 && result2 == 2);
}

int main() {

    test_basic_defer();
    test_multiple_defers();
    test_defer_with_variables();
    test_defer_in_function();
    test_nested_scopes();
    test_defer_with_complex_block();
    test_defer_unique_names();

    return 0;
}