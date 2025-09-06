DOCKER_RUN = docker run --rm -v $(PWD):/workspace main sh -c

.PHONY: build-image
build-image:
	docker build -t main .

.PHONY: build
build:
	$(DOCKER_RUN) 'rm -rf build && mkdir -p build && cd build && cmake .. && cmake --build . -j$$(nproc)'

.PHONY: run
run: build
	$(DOCKER_RUN) 'cd build && ./binary'

.PHONY: lint
lint: build
# 	$(DOCKER_RUN) 'cppcheck --std=c11 --enable=information --suppress=missingInclude -I src src'
# 	$(DOCKER_RUN) 'cd build && find ../src -name "*.c" | xargs -r clang-tidy -p . --header-filter=.* --checks=-*,readability-*,performance-*,bugprone-*'	
	$(DOCKER_RUN) 'cd build && find ../src -name "*.c" -o -name "*.h" | xargs -r include-what-you-use -p .'

.PHONY: memcheck
memcheck: build
	$(DOCKER_RUN) 'cd build && valgrind --leak-check=full ./binary'

.PHONY: test
test: build
	$(DOCKER_RUN) 'cd build && ctest --verbose'

.PHONY: fmt
fmt:
	$(DOCKER_RUN) 'find . -name "*.c" -o -name "*.h" | xargs clang-format -i'

.PHONY: vuln-scan
vuln-scan: build
	$(DOCKER_RUN) 'trivy fs --security-checks vuln --format table .'
	$(DOCKER_RUN) 'cd build && cve-bin-tool ./binary --format console'
	$(DOCKER_RUN) 'lynis audit system --quick'

.PHONY: vuln-scan-json
vuln-scan-json: build
	@echo "Running vulnerability scans with JSON output..."
	$(DOCKER_RUN) 'trivy fs --security-checks vuln --format json --output vuln-report.json .'
	$(DOCKER_RUN) 'cd build && cve-bin-tool ./binary --format json --output-file ../binary-vuln-report.json'
	@echo "Reports saved as vuln-report.json and binary-vuln-report.json"

.PHONY: clean
clean:
	docker rmi main 2>/dev/null || true
	rm -rf build
	rm -f vuln-report.json binary-vuln-report.json
