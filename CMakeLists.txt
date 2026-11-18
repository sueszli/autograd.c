cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(autograd LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

# no fma contraction: parity with the float64 reference depends on exact rounding
add_compile_options(-Wall -Wextra -Wconversion -Wsign-conversion -Werror -ffp-contract=off)

if(NOT DISABLE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

include(FetchContent)
FetchContent_Declare(unity GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git GIT_TAG v2.6.1 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(unity)

file(GLOB_RECURSE SRCS src/*.c)
add_library(lib ${SRCS})
target_include_directories(lib PUBLIC src)
target_link_libraries(lib PUBLIC m)

enable_testing()
file(GLOB TEST_FILES test/*.c)
foreach(TEST_FILE ${TEST_FILES})
  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
  add_executable(${TEST_NAME} ${TEST_FILE})
  target_link_libraries(${TEST_NAME} PRIVATE unity lib)
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
