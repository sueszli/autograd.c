cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(autograd LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

# no fma contraction, so results are identical across compilers and cpus
add_compile_options(-Wall -Wextra -Wconversion -Wsign-conversion -Werror -ffp-contract=off)

if(NOT DISABLE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

include(FetchContent)
FetchContent_Declare(unity GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git GIT_TAG v2.6.1 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(unity)
target_compile_definitions(unity PUBLIC UNITY_INCLUDE_DOUBLE)
# download only, SOURCE_SUBDIR points nowhere so its own CMakeLists.txt is skipped
FetchContent_Declare(cjson GIT_REPOSITORY https://github.com/DaveGamble/cJSON.git GIT_TAG v1.7.18 GIT_SHALLOW TRUE SOURCE_SUBDIR none)
FetchContent_MakeAvailable(cjson)

file(GLOB_RECURSE SRCS src/*.c)
list(FILTER SRCS EXCLUDE REGEX "src/parity\\.c$")
add_library(lib ${SRCS})
target_include_directories(lib PUBLIC src)
target_link_libraries(lib PUBLIC m)

# cJSON is third-party code, so it is built without our warning flags
set_source_files_properties(${cjson_SOURCE_DIR}/cJSON.c PROPERTIES COMPILE_OPTIONS "-w")
add_executable(parity src/parity.c ${cjson_SOURCE_DIR}/cJSON.c)
target_include_directories(parity PRIVATE ${cjson_SOURCE_DIR})
target_link_libraries(parity PRIVATE lib)

enable_testing()
file(GLOB TEST_FILES test/*.c)
foreach(TEST_FILE ${TEST_FILES})
  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
  add_executable(${TEST_NAME} ${TEST_FILE})
  target_link_libraries(${TEST_NAME} PRIVATE unity lib)
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
add_test(NAME parity COMMAND parity WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
