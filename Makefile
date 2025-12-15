.PHONY: run
run: lint
	mkdir -p /tmp/build && cd /tmp/build && cmake -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang $(PWD) && cmake --build . -j$$(sysctl -n hw.ncpu) && cd $(PWD) && ASAN_OPTIONS=detect_leaks=1 LSAN_OPTIONS=suppressions=$(PWD)/suppr.txt /tmp/build/binary

.PHONY: leaks
leaks: lint
	rm -rf /tmp/leaks-build && mkdir -p /tmp/leaks-build && cd /tmp/leaks-build && cmake -DDISABLE_ASAN=ON $(PWD) && cmake --build . -j$$(sysctl -n hw.ncpu)
	codesign -s - -f --entitlements entitlements.plist /tmp/leaks-build/binary
	cd $(PWD) && leaks --atExit --list --groupByType -- /tmp/leaks-build/binary

.PHONY: test
test: lint
	rm -rf /tmp/test-build && mkdir -p /tmp/test-build && cd /tmp/test-build && cmake -DBUILD_TESTS=ON $(PWD) && cmake --build . -j$$(sysctl -n hw.ncpu) && ctest --output-on-failure

.PHONY: run-release
run-release:
	rm -rf /tmp/release-build && mkdir -p /tmp/release-build && cd /tmp/release-build && cmake -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang -DCMAKE_BUILD_TYPE=Release -DDISABLE_ASAN=ON -DDISABLE_UBSAN=ON -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON $(PWD) && cmake --build . -j$$(sysctl -n hw.ncpu) && cd $(PWD) && /tmp/release-build/binary

# 
# utils
# 

.PHONY: download
download:
	test ! -f data/data_batch_1.bin # check if already downloaded
	mkdir -p data
	test -f data/cifar-10-binary.tar.gz || wget -O data/cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
	tar -xzf data/cifar-10-binary.tar.gz -C data --strip-components=1

.PHONY: lint
lint:
	cppcheck --enable=all --std=c23 --language=c --suppressions-list=cppcheck-suppressions.txt --check-level=exhaustive --inconclusive --inline-suppr -I src/ src/

.PHONY: fmt
fmt:
	uvx --from cmakelang cmake-format --dangle-parens --line-width 500 -i CMakeLists.txt
	find . -name "*.c" -o -name "*.h" | xargs clang-format -i
