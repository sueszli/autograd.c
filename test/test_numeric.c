#include "std/numeric.h"
#include "unity.h"
#include <math.h>

void setUp(void) {}
void tearDown(void) {}

void test_max_abs_diff(void) {
    TEST_ASSERT_EQUAL_DOUBLE(2, max_abs_diff((double[]){0, 3}, (double[]){1, 1}, 2));
    TEST_ASSERT_EQUAL_DOUBLE(INFINITY, max_abs_diff((double[]){0, NAN}, (double[]){0, 1}, 2));
}

static double cube(void *x) { return pow(*(double *)x, 3); }

void test_central_difference(void) {
    double x = 2;
    TEST_ASSERT_DOUBLE_WITHIN(1e-6, 12, central_difference(cube, &x, &x, 1e-4));
    TEST_ASSERT_EQUAL_DOUBLE(2, x);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_max_abs_diff);
    RUN_TEST(test_central_difference);
    return UNITY_END();
}
