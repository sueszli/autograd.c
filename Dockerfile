FROM --platform=linux/amd64 ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential gcc g++ clang cmake make clang-tidy clang-format-14 cppcheck valgrind git curl bash iwyu python3 python3-pip lynis wget \
    && ln -s /usr/bin/clang-format-14 /usr/bin/clang-format \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir cve-bin-tool

RUN curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

WORKDIR /workspace