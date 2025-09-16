DOCKER_RUN = docker run --rm -v $(PWD):/workspace main sh -c

.PHONY: download
download:
	mkdir -p data
	test -f data/cifar-10-binary.tar.gz || wget -O data/cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
	tar -xzf data/cifar-10-binary.tar.gz -C data --strip-components=1

.PHONY: build-image
build-image:
	docker build -t main .

.PHONY: run
run:
	$(DOCKER_RUN) "cd $(mktemp -d) && cmake /workspace && make -j$(nproc) && ./autograd"

.PHONY: fmt
fmt:
	$(DOCKER_RUN) 'find . -name "*.c" -o -name "*.h" | xargs clang-format -i'

.PHONY: clean
clean:
	docker rmi main
