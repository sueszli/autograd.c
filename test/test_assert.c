#include "std/assert.h"
#include "unity.h"
#include <math.h>

void setUp(void) {}
void tearDown(void) {}

void test_assert_allclose_returns_max_abs_diff(void) {
    TEST_ASSERT_EQUAL_DOUBLE(2, assert_allclose((f64[]){0, 3}, (f64[]){1, 1}, 2, 0.0, 2.0, "x"));
    // rtol scales with |desired|, 0.5 <= 0 + 0.1 * 10
    TEST_ASSERT_EQUAL_DOUBLE(0.5, assert_allclose((f64[]){10.5}, (f64[]){10}, 1, 0.1, 0.0, "x"));
}

static f64 cube(void *x) { return pow(*(f64 *)x, 3); }

void test_assert_grad_restores_x(void) {
    f64 x = 2;
    assert_grad(cube, &x, &x, (f64[]){12}, 1, 0.0, 1e-6, "d/dx x^3 at 2");
    TEST_ASSERT_EQUAL_DOUBLE(2, x);
}

i32 main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_assert_allclose_returns_max_abs_diff);
    RUN_TEST(test_assert_grad_restores_x);
    return UNITY_END();
}
