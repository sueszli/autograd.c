cmake_minimum_required(VERSION 3.16)
project(main LANGUAGES C)

# set(CMAKE_REQUIRED_QUIET ON)

option(BUILD_TESTS "Test mode" OFF)
option(DISABLE_ASAN "Disable AddressSanitizer" OFF)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_FLAGS "-g -O0")

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra
        # -Wpedantic -Wmissing-prototypes -Wtrampolines -Wshadow
        -Wcast-align -Wcast-qual -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-include-dirs -Wredundant-decls -Wstrict-overflow=5 -Wundef -Wuninitialized -Wwrite-strings -Wpointer-arith -Wstrict-prototypes -Wconversion -Wsign-conversion -Wbad-function-cast -Wjump-misses-init -Wold-style-definition -Wmissing-parameter-type -Wempty-body -Wtype-limits -Wclobbered -Wunused-but-set-parameter -Wunused-but-set-variable -Wmaybe-uninitialized -Wstack-protector -fstack-protector-strong -fno-common -Wformat-security -Wno-implicit-function-declaration
    )

    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        add_compile_options(
            -Wduplicated-cond -Wduplicated-branches -Wnull-dereference -Walloc-zero -Walloca -Wstringop-overflow=4 -Warray-bounds=2
        )
    endif()

    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        add_compile_options(
            -Wcomma -Wloop-analysis -Wshift-sign-overflow -Wconditional-uninitialized -Wdocumentation
        )
    endif()
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
    set(ENABLE_CJSON_TEST OFF CACHE BOOL "" FORCE)
    set(CJSON_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(UNITY_EXTENSION_FIXTURE OFF CACHE BOOL "" FORCE)
    set(UNITY_EXTENSION_MEMORY OFF CACHE BOOL "" FORCE)
    set(UNITY_BUILD_TESTS OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(unity)
endif()

#
# source
#

file(GLOB UTIL_SOURCES "src/utils/*.c")
add_library(utils STATIC ${UTIL_SOURCES})
target_include_directories(utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB MAIN_SOURCES "src/*.c")
file(GLOB DATASET_SOURCES "src/datasets/*.c")
file(GLOB MATH_SOURCES "src/math/*.c")
add_executable(binary ${MAIN_SOURCES} ${DATASET_SOURCES} ${MATH_SOURCES})
target_link_libraries(binary PRIVATE utils m) # math
target_include_directories(binary PRIVATE ${cjson_SOURCE_DIR})

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang" AND NOT DISABLE_ASAN)
    target_compile_options(binary PRIVATE -fsanitize=address)
    target_link_options(binary PRIVATE -fsanitize=address)
endif()

# 
# tests
# 

if(BUILD_TESTS)
    enable_testing()

    file(GLOB TEST_SOURCES "test/*.c")

    foreach(TEST_SOURCE ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
        set(TEST_EXECUTABLE "${TEST_NAME}_binary")
        
        add_executable(${TEST_EXECUTABLE} ${TEST_SOURCE})
        target_link_libraries(${TEST_EXECUTABLE} PRIVATE unity utils)
        target_include_directories(${TEST_EXECUTABLE} PRIVATE ${unity_SOURCE_DIR}/src)
        
        if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang" AND NOT DISABLE_ASAN)
            target_compile_options(${TEST_EXECUTABLE} PRIVATE -fsanitize=address)
            target_link_options(${TEST_EXECUTABLE} PRIVATE -fsanitize=address)
        endif()
        
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_EXECUTABLE})
    endforeach()
endif()
