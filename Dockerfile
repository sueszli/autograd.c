FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential gcc g++ clang cmake make clang-tidy clang-format-14 cppcheck valgrind git curl bash \
    && ln -s /usr/bin/clang-format-14 /usr/bin/clang-format \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
