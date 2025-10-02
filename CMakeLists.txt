cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(defer LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(CMAKE_C_COMPILER_ID MATCHES "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 14.0.0)
  message(FATAL_ERROR "GCC is outdated: ${CMAKE_C_COMPILER_VERSION}")
endif()
if(CMAKE_C_COMPILER_ID MATCHES "Clang" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 16.0.0)
  message(FATAL_ERROR "Clang is outdated: ${CMAKE_C_COMPILER_VERSION}")
endif()

#
# compiler hardening
#
# see: https://best.openssf.org/Compiler-Hardening-Guides/Compiler-Options-Hardening-Guide-for-C-and-C++
#

add_compile_options(
  -O2
  -Wall
  -Wextra
  -Wformat
  -Wformat=2
  -Wconversion
  -Wsign-conversion
  -Wimplicit-fallthrough
  -Werror=format-security
  # more portable and explicit than -fhardened
  -U_FORTIFY_SOURCE
  -D_FORTIFY_SOURCE=3
  -D_GLIBCXX_ASSERTIONS
  -fstrict-flex-arrays=3
  -fstack-protector-strong
  # deprecated c calls
  -Werror=implicit
  -Werror=incompatible-pointer-types
  -Werror=int-conversion
  # multithreading with pthreads
  -fexceptions
  # for shared libraries use `-fPIC`
  -fPIE
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-fno-delete-null-pointer-checks -fno-strict-overflow -fno-strict-aliasing -ftrivial-auto-var-init=zero)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_compile_options(-fstack-clash-protection)
  add_link_options(
    -pie
    -Wl,-z,nodlopen
    -Wl,-z,noexecstack
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,--as-needed
    -Wl,--no-copy-dt-needed-entries
  )
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU")
  # add `-fzero-init-padding-bits=all` from gcc 15
  add_compile_options(-Wtrampolines -Wbidi-chars=any)
endif()
if(CMAKE_C_COMPILER_ID MATCHES "Clang")
  # nop
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686|i386")
  add_compile_options(-fcf-protection=full)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  add_compile_options(-mbranch-protection=standard)
endif()

#
# sanitizers
#

if(NOT DISABLE_ASAN)
  # apple silicon requires clang from homebrew with paths set accordingly: `brew info llvm`
  add_compile_options(-fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address)
endif()

if(NOT DISABLE_UBSAN)
  add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=undefined)
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

  set(BUILD_TESTING
      OFF
      CACHE BOOL "" FORCE
  )
  set(BUILD_SHARED_LIBS
      OFF
      CACHE BOOL "" FORCE
  )
  set(UNITY_EXTENSION_FIXTURE
      OFF
      CACHE BOOL "" FORCE
  )
  set(UNITY_EXTENSION_MEMORY
      OFF
      CACHE BOOL "" FORCE
  )
  set(UNITY_BUILD_TESTS
      OFF
      CACHE BOOL "" FORCE
  )

  FetchContent_MakeAvailable(unity)
endif()

#
# sources
#

add_executable(binary src/main.c)

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
    target_link_libraries(${TEST_EXECUTABLE} PRIVATE unity)
    target_include_directories(${TEST_EXECUTABLE} PRIVATE ${unity_SOURCE_DIR}/src)

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_EXECUTABLE})
  endforeach()
endif()
