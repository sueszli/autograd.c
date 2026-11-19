#include "std/json.h"
#include "unity.h"

void setUp(void) {}
void tearDown(void) {}

void test_matrix_and_its_transpose(void) {
    cJSON *j = cJSON_Parse("[[1, 2, 3], [4, 5, 6]]");
    f64 out[6];
    json_matrix(j, out, 2, 3, false);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((f64[]){1, 2, 3, 4, 5, 6}), out, 6);
    json_matrix(j, out, 3, 2, true);
    TEST_ASSERT_EQUAL_DOUBLE_ARRAY(((f64[]){1, 4, 2, 5, 3, 6}), out, 6);
    cJSON_Delete(j);
}

void test_get(void) {
    cJSON *j = cJSON_Parse("{\"bos\": 26}");
    TEST_ASSERT_EQUAL_DOUBLE(26, json_get(j, "bos")->valuedouble);
    cJSON_Delete(j);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_matrix_and_its_transpose);
    RUN_TEST(test_get);
    return UNITY_END();
}
