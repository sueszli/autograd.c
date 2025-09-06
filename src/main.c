#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("whatever\n");

    int32_t count = 42;
    float pi = 3.14f;

    printf("Integer: %d\n", count);
    printf("Float: %.2f\n", pi);

    const char *greeting = "Hello!";
    printf("String: %s\n", greeting);

    int numbers[] = {1, 2, 3};

    printf("Array: ");
    for (size_t i = 0; i < 3; i++) {
        printf("%d ", numbers[i]);
    }
    printf("\n");

    return EXIT_SUCCESS;
}
