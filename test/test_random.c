#include "std/random.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_same_seed_gives_same_numbers(void) {
    uint64_t a = 7, b = 7;
    for (int i = 0; i < 100; i++) {
        double x = random_float(&a);
        TEST_ASSERT_EQUAL_DOUBLE(x, random_float(&b));
        TEST_ASSERT_TRUE(x >= 0.0 && x < 1.0);
    }
}

void test_uniform_stays_in_range(void) {
    uint64_t s = 1;
    double out[1000];
    random_uniform(&s, out, 1000, -0.3, 0.3);
    for (int i = 0; i < 1000; i++) {
        TEST_ASSERT_TRUE(out[i] >= -0.3 && out[i] < 0.3);
    }
}

void test_choice_follows_p(void) {
    uint64_t s = 1;
    TEST_ASSERT_EQUAL_INT64(2, random_choice(&s, (double[]){0, 0, 1}, 3));
    TEST_ASSERT_EQUAL_INT64(0, random_choice(&s, (double[]){1, 0, 0}, 3));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_same_seed_gives_same_numbers);
    RUN_TEST(test_uniform_stays_in_range);
    RUN_TEST(test_choice_follows_p);
    return UNITY_END();
}
