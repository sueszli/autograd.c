cmake_minimum_required(VERSION 3.16)
project(main LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra
        # -Wpedantic -Wcast-align -Wcast-qual -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wredundant-decls -Wshadow -Wstrict-overflow=5 -Wundef -Wuninitialized -Wwrite-strings -Wpointer-arith -Wmissing-prototypes -Wstrict-prototypes -Wconversion -Wsign-conversion -Wbad-function-cast -Wc++-compat -Wjump-misses-init -Wold-style-definition -Wmissing-parameter-type -Wempty-body -Wtype-limits -Wclobbered -Wunused-but-set-parameter -Wunused-but-set-variable -Wmaybe-uninitialized -Wstack-protector -fstack-protector-strong -fno-common -Wformat-security -Wtrampolines
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

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3")

# 
# dependencies
# 

include(FetchContent)

FetchContent_Declare(
    cjson
    GIT_REPOSITORY https://github.com/DaveGamble/cJSON.git
    GIT_TAG v1.7.18
    GIT_SHALLOW TRUE
)


FetchContent_Declare(
    unity
    GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
    GIT_TAG v2.6.1
    GIT_SHALLOW TRUE
)

set(ENABLE_CJSON_TEST OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(cjson unity)

# 
# binary
# 

add_executable(binary src/main.c)
target_link_libraries(binary PRIVATE cjson unity)
target_include_directories(binary PRIVATE ${cjson_SOURCE_DIR} ${unity_SOURCE_DIR}/src)

if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(binary PRIVATE -fsanitize=address)
    target_link_options(binary PRIVATE -fsanitize=address)
endif()
