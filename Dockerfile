FROM --platform=linux/amd64 ubuntu:22.04

RUN apt-get update && apt-get install -y \
    gcc g++ cmake make valgrind clang-format-14
RUN ln -s /usr/bin/clang-format-14 /usr/bin/clang-format
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace
