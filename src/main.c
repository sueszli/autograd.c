#include <cJSON.h>
#include <stdatomic.h>
#include <stdio.h>
#include <time.h>
#include <unity.h>

#include "go.h"

#define NUM_TASKS 4

static void do_work(int task_id) {
    int result = 0;
    for (int i = 0; i < 1000000000; i++) {
        result += i % 7;
    }
    printf("Task %d done (result: %d)\n", task_id, result);
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

void setUp(void) {
}

void tearDown(void) {
}

void test_cjson_functionality(void) {
    cJSON *json = cJSON_CreateObject();
    cJSON *name = cJSON_CreateString("Test Project");
    cJSON_AddItemToObject(json, "name", name);
    
    TEST_ASSERT_NOT_NULL(json);
    cJSON *retrieved_name = cJSON_GetObjectItem(json, "name");
    TEST_ASSERT_NOT_NULL(retrieved_name);
    TEST_ASSERT_EQUAL_STRING("Test Project", cJSON_GetStringValue(retrieved_name));
    
    cJSON_Delete(json);
}

void test_time_function(void) {
    double time1 = get_time_ms();
    double time2 = get_time_ms();
    TEST_ASSERT_TRUE(time2 >= time1);
}

int main(void) {
    UNITY_BEGIN();
    
    RUN_TEST(test_cjson_functionality);
    RUN_TEST(test_time_function);
    
    int test_result = UNITY_END();
    
    if (test_result == 0) {
        printf("\nAll tests passed! Running main application:\n\n");
        
        cJSON *json = cJSON_CreateObject();
        cJSON *name = cJSON_CreateString("Modern C Project");
        cJSON *version = cJSON_CreateString("1.0.0");
        cJSON_AddItemToObject(json, "name", name);
        cJSON_AddItemToObject(json, "version", version);

        char *json_string = cJSON_Print(json);
        printf("JSON: %s\n\n", json_string);

        cJSON_Delete(json);
        free(json_string);

        double start = get_time_ms();
        for (int i = 1; i <= NUM_TASKS; i++) {
            do_work(i);
        }
        double sync_time = get_time_ms() - start;
        printf("Synchronous total time: %.0f ms\n\n", sync_time);

        start = get_time_ms();
        for (int i = 1; i <= NUM_TASKS; i++) {
            go({ do_work(i); });
        }
        wait();
        double concurrent_time = get_time_ms() - start;
        printf("Concurrent total time: %.0f ms\n\n", concurrent_time);

        printf("Speedup: %.1fx faster with concurrent approach\n", sync_time / concurrent_time);
    }

    return test_result;
}
