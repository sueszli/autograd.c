#include "std/testing.h"
#include "unity.h"
#include <math.h>

void setUp(void) {}
void tearDown(void) {}

void test_assert_allclose_returns_max_abs_diff(void) {
    TEST_ASSERT_EQUAL_DOUBLE(2, assert_allclose((double[]){0, 3}, (double[]){1, 1}, 2, 0.0, 2.0, "x"));
    // rtol scales with |desired|: 0.5 <= 0 + 0.1 * 10
    TEST_ASSERT_EQUAL_DOUBLE(0.5, assert_allclose((double[]){10.5}, (double[]){10}, 1, 0.1, 0.0, "x"));
}

static double cube(void *x) { return pow(*(double *)x, 3); }

void test_central_difference(void) {
    double x = 2;
    assert_close(central_difference(cube, &x, &x, 1e-4), 12, 0.0, 1e-6, "d/dx x^3 at 2");
    TEST_ASSERT_EQUAL_DOUBLE(2, x);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_assert_allclose_returns_max_abs_diff);
    RUN_TEST(test_central_difference);
    return UNITY_END();
}
