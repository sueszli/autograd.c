DOCKER_RUN = docker run --rm -v $(PWD):/workspace c-image sh -c

.PHONY: build
build:
	docker build -t c-image .
	$(DOCKER_RUN) 'rm -rf build && mkdir -p build && cd build && cmake .. && make -j$$(nproc)'

.PHONY: fmt
fmt:
	$(DOCKER_RUN) 'find . -name "*.c" -o -name "*.h" | xargs clang-format -i'

.PHONY: check
check:
	$(DOCKER_RUN) 'cppcheck --std=c23 . && clang-tidy src/main.c -- -std=c23'
	$(DOCKER_RUN) 'cd build && valgrind --leak-check=full ./binary'

.PHONY: run
run:
	$(DOCKER_RUN) 'cd build && ./binary'

.PHONY: clean
clean:
	docker rmi c-image 2>/dev/null || true
	rm -rf build
