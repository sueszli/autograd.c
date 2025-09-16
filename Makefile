DOCKER_RUN = docker run --rm -v $(PWD):/workspace main sh -c

.PHONY: download
download:
	mkdir -p data
	test -f data/cifar-10-binary.tar.gz || wget -O data/cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
	tar -xzf data/cifar-10-binary.tar.gz -C data --strip-components=1

.PHONY: build-image
build-image:
	docker build -t main .

# .PHONY: run
# run:
# 	$(DOCKER_RUN) 'rm -rf build && mkdir -p build && cd build && cmake .. && cmake --build . -j$(nproc)'
# 	$(DOCKER_RUN) 'cd build && ./binary'

# .PHONY: memcheck
# memcheck:
# 	$(DOCKER_RUN) 'rm -rf build && mkdir -p build && cd build && cmake -DDISABLE_ASAN=ON .. && cmake --build . -j$(nproc)'
# 	$(DOCKER_RUN) 'cd build && valgrind --leak-check=full ./binary'

# .PHONY: test
# test:
# 	$(DOCKER_RUN) 'rm -rf build && mkdir -p build && cd build && cmake -DBUILD_TESTS=ON .. && cmake --build . -j$(nproc)'
# 	$(DOCKER_RUN) 'cd build && ctest --verbose'

# .PHONY: fmt
# fmt:
# 	$(DOCKER_RUN) 'find . -name "*.c" -o -name "*.h" | xargs clang-format -i'

.PHONY: fmt
fmt:
	$(DOCKER_RUN) 'find . -name "*.c" -o -name "*.h" | xargs clang-format -i'

.PHONY: clean
clean:
	docker rmi main 2>/dev/null || true
	rm -rf build
