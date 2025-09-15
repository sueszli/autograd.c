cmake_minimum_required(VERSION 3.16)
project(main LANGUAGES C)

# 
# arguments
# 

option(BUILD_TESTS "Test mode" OFF)
option(DISABLE_ASAN "Disable AddressSanitizer" OFF)

# 
# compiler settings
# 

if(NOT (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang"))
    message(FATAL_ERROR "Using unsupported compiler: ${CMAKE_C_COMPILER_ID}")
endif()

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_FLAGS "-g -O0")
if(NOT DISABLE_ASAN)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
endif()

add_compile_options(
    -Wall -Wextra
    # -Wpedantic -Wmissing-prototypes -Wtrampolines -Wshadow
    -Wcast-align -Wcast-qual -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-include-dirs -Wredundant-decls -Wstrict-overflow=5 -Wundef -Wuninitialized -Wwrite-strings -Wpointer-arith -Wstrict-prototypes -Wconversion -Wsign-conversion -Wbad-function-cast -Wjump-misses-init -Wold-style-definition -Wmissing-parameter-type -Wempty-body -Wtype-limits -Wclobbered -Wunused-but-set-parameter -Wunused-but-set-variable -Wmaybe-uninitialized -Wstack-protector -fstack-protector-strong -fno-common -Wformat-security -Wno-implicit-function-declaration
)

if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wduplicated-cond -Wduplicated-branches -Wnull-dereference -Walloc-zero -Walloca -Wstringop-overflow=4 -Warray-bounds=2)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wcomma -Wloop-analysis -Wshift-sign-overflow -Wconditional-uninitialized -Wdocumentation)
endif()

# 
# dependencies
# 

include(FetchContent)

if(BUILD_TESTS)
    FetchContent_Declare(
        unity
        GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
        GIT_TAG v2.6.1
        GIT_SHALLOW TRUE
    )

    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(UNITY_EXTENSION_FIXTURE OFF CACHE BOOL "" FORCE)
    set(UNITY_EXTENSION_MEMORY OFF CACHE BOOL "" FORCE)
    set(UNITY_BUILD_TESTS OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(unity)
endif()

#
# source
#

file(GLOB UTILS_SRC src/utils/*.c)
add_library(utils STATIC ${UTILS_SRC})
target_include_directories(utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB TENSOR_SRC src/tensor/*.c)
add_library(tensor STATIC ${TENSOR_SRC})
target_include_directories(tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB DATASETS_SRC src/datasets/*.c)
add_library(datasets STATIC ${DATASETS_SRC})
target_include_directories(datasets INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB EVAL_SRC src/eval/*.c)
add_library(eval STATIC ${EVAL_SRC})
target_include_directories(eval INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(eval PRIVATE datasets)

file(GLOB BINARY_SRC src/main.c)
add_executable(binary ${BINARY_SRC})
target_link_libraries(binary PRIVATE utils tensor datasets eval m)

# 
# tests
# 

file(GLOB_RECURSE TEST_FILES test/*.c)

if(BUILD_TESTS)
    enable_testing()

    foreach(TEST_FILE ${TEST_FILES})
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
        set(TEST_SOURCE "test/${TEST_NAME}.c")
        set(TEST_EXECUTABLE "${TEST_NAME}_binary")

        add_executable(${TEST_EXECUTABLE} ${TEST_SOURCE})
        target_link_libraries(${TEST_EXECUTABLE} PRIVATE unity utils)
        target_include_directories(${TEST_EXECUTABLE} PRIVATE ${unity_SOURCE_DIR}/src)

        # some tests need additional source files
        if (TEST_NAME STREQUAL "test_metrics")
            target_link_libraries(${TEST_EXECUTABLE} PRIVATE datasets eval)
        endif()
        
        if (TEST_NAME STREQUAL "test_reshape" OR TEST_NAME STREQUAL "test_ops")
            target_link_libraries(${TEST_EXECUTABLE} PRIVATE tensor m)
        endif()

        add_test(NAME ${TEST_NAME} COMMAND ${TEST_EXECUTABLE})
    endforeach()
endif()
