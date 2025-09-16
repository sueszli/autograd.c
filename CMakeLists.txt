cmake_minimum_required(VERSION 3.20)
project(autograd LANGUAGE C)

if(NOT CMAKE_C_COMPILER_ID MATCHES "GNU") message(FATAL_ERROR "needs gcc.") endif()
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

# enable `-DCMAKE_BUILD_TYPE=Debug` by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Enable all warnings and make them errors
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpedantic -Wconversion -Wsign-conversion")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wcast-align -Wcast-qual -Wdisabled-optimization")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wfloat-equal -Wformat=2 -Wformat-nonliteral")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wformat-security -Wformat-y2k -Wimport")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Winit-self -Winline -Winvalid-pch")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wlong-long -Wmissing-field-initializers")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmissing-format-attribute -Wmissing-include-dirs")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmissing-noreturn -Wpacked -Wpadded")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpointer-arith -Wredundant-decls -Wshadow")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wstack-protector -Wstrict-aliasing=2")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wswitch-default -Wswitch-enum")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wunreachable-code -Wunused -Wunused-parameter")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wvariadic-macros -Wwrite-strings")

# GCC-specific warnings (most pedantic possible)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wlogical-op -Wstrict-overflow=5")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wundef -Wjump-misses-init")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wdouble-promotion -Wformat-truncation=2")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wformat-overflow=2 -Wstringop-overflow=4")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Walloc-zero -Walloca -Warray-bounds=2")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wattribute-alias=2 -Wduplicated-branches")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wduplicated-cond -Wtrampolines")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wfloat-conversion -Wmissing-prototypes")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wnested-externs -Wold-style-definition")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wstrict-prototypes -Wsuggest-attribute=cold")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wsuggest-attribute=const -Wsuggest-attribute=malloc")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wsuggest-attribute=noreturn -Wsuggest-attribute=pure")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-aggressive-loop-optimizations")

# Sanitizers (compatible combination for maximum detection)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O1 -g -fno-omit-frame-pointer")

# AddressSanitizer + UndefinedBehaviorSanitizer
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=undefined")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=shift")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=shift-exponent")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=shift-base")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=integer-divide-by-zero")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=unreachable")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=vla-bound")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=null")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=return")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=signed-integer-overflow")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=bounds")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=bounds-strict")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=alignment")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=object-size")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=float-divide-by-zero")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=float-cast-overflow")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=nonnull-attribute")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=returns-nonnull-attribute")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=bool")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=enum")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=pointer-overflow")

# Additional sanitizer options for maximum pedantry
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize-address-use-after-scope")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-sanitize-recover=all")  # Exit on first error
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fstack-check")

# Stack protection
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstack-protector-strong")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstack-clash-protection")

# Link flags for sanitizers
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=undefined")

# Create executable
add_executable(autograd src/main.c)

# ThreadSanitizer (incompatible with AddressSanitizer)
add_executable(autograd_tsan src/main.c)
target_compile_options(autograd_tsan PRIVATE
    -fsanitize=thread
    -fno-omit-frame-pointer
    -g -O1
)
target_link_options(autograd_tsan PRIVATE -fsanitize=thread)

# LeakSanitizer standalone
add_executable(autograd_lsan src/main.c)
target_compile_options(autograd_lsan PRIVATE
    -fsanitize=leak
    -fno-omit-frame-pointer
    -g -O1
)
target_link_options(autograd_lsan PRIVATE -fsanitize=leak)

# Print build configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER_ID}")
message(STATUS "C Flags: ${CMAKE_C_FLAGS}")
message(STATUS "C Debug Flags: ${CMAKE_C_FLAGS_DEBUG}")
