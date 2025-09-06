#include <cJSON.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unity.h>

void setUp(void) {
}

void tearDown(void) {
}

void test_basic_math(void) {
    TEST_ASSERT_EQUAL(4, 2 + 2);
    TEST_ASSERT_EQUAL(0, 5 - 5);
    TEST_ASSERT_EQUAL(15, 3 * 5);
    TEST_ASSERT_EQUAL(2, 10 / 5);
}

void test_string_operations(void) {
    char str1[] = "Hello";
    char str2[] = "Hello";
    char str3[] = "World";
    
    TEST_ASSERT_EQUAL_STRING(str1, str2);
    TEST_ASSERT_NOT_EQUAL(strcmp(str1, str3), 0);
}

void test_cjson_functionality(void) {
    cJSON *json = cJSON_CreateObject();
    cJSON *name = cJSON_CreateString("Test Project");
    cJSON *version = cJSON_CreateNumber(1.0);
    
    cJSON_AddItemToObject(json, "name", name);
    cJSON_AddItemToObject(json, "version", version);
    
    TEST_ASSERT_NOT_NULL(json);
    
    cJSON *retrieved_name = cJSON_GetObjectItem(json, "name");
    TEST_ASSERT_NOT_NULL(retrieved_name);
    TEST_ASSERT_EQUAL_STRING("Test Project", cJSON_GetStringValue(retrieved_name));
    
    cJSON *retrieved_version = cJSON_GetObjectItem(json, "version");
    TEST_ASSERT_NOT_NULL(retrieved_version);
    TEST_ASSERT_EQUAL_FLOAT(1.0, (float)cJSON_GetNumberValue(retrieved_version));
    
    cJSON_Delete(json);
}

void test_array_operations(void) {
    int arr[] = {1, 2, 3, 4, 5};
    int size = sizeof(arr) / sizeof(arr[0]);
    
    TEST_ASSERT_EQUAL(5, size);
    TEST_ASSERT_EQUAL(1, arr[0]);
    TEST_ASSERT_EQUAL(5, arr[4]);
    
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    TEST_ASSERT_EQUAL(15, sum);
}

void test_time_functionality(void) {
    struct timespec ts;
    int result = clock_gettime(CLOCK_MONOTONIC, &ts);
    
    TEST_ASSERT_EQUAL(0, result);
    TEST_ASSERT_TRUE(ts.tv_sec > 0);
    TEST_ASSERT_TRUE(ts.tv_nsec >= 0);
    TEST_ASSERT_TRUE(ts.tv_nsec < 1000000000);
}

void test_atomic_operations(void) {
    atomic_int counter = ATOMIC_VAR_INIT(0);
    
    atomic_fetch_add(&counter, 5);
    TEST_ASSERT_EQUAL(5, atomic_load(&counter));
    
    atomic_fetch_sub(&counter, 2);
    TEST_ASSERT_EQUAL(3, atomic_load(&counter));
    
    atomic_store(&counter, 10);
    TEST_ASSERT_EQUAL(10, atomic_load(&counter));
}

int main(void) {
    UNITY_BEGIN();
    
    RUN_TEST(test_basic_math);
    RUN_TEST(test_string_operations);
    RUN_TEST(test_cjson_functionality);
    RUN_TEST(test_array_operations);
    RUN_TEST(test_time_functionality);
    RUN_TEST(test_atomic_operations);
    
    return UNITY_END();
}