NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu)

.PHONY: test
test:
	cmake -S . -B build/test -DCMAKE_BUILD_TYPE=Debug && cmake --build build/test -j$(NPROC) && ctest --test-dir build/test --output-on-failure

.PHONY: lint
lint:
	cppcheck --enable=all --std=c23 --language=c --suppressions-list=cppcheck-suppressions.txt --check-level=exhaustive --inconclusive --inline-suppr -I src/ src/

.PHONY: fmt
fmt:
	find src test -name "*.c" -o -name "*.h" | xargs clang-format -i
