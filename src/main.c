#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "benchmark.h"

int main(void) {
    printf("Hello, World!\n");

    benchmark("sleep benchmark", { sleep(4); });

    return EXIT_SUCCESS;
}