NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu)

.PHONY: test
test:
	cmake -S . -B build/test -DCMAKE_BUILD_TYPE=Debug && cmake --build build/test -j$(NPROC) && ctest --test-dir build/test --output-on-failure

.PHONY: parity
parity:
	cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DDISABLE_SANITIZERS=ON && cmake --build build/release -j$(NPROC) && ./build/release/parity

.PHONY: lint
lint:
	cppcheck --enable=all --std=c23 --language=c --suppressions-list=cppcheck-suppressions.txt --check-level=exhaustive --inconclusive --inline-suppr -I src/ src/

.PHONY: fmt
fmt:
	find src test -name "*.c" -o -name "*.h" | xargs clang-format -i

.PHONY: precommit-hook
precommit-hook:
	@common_dir="$$(git rev-parse --git-common-dir 2>/dev/null)"; \
	hooks_path="$$(git config --get core.hooksPath 2>/dev/null)"; \
	hook="$$common_dir/hooks/pre-commit"; \
	if [ -n "$$common_dir" ] && [ -z "$$hooks_path" ] && [ ! -e "$$hook" ] && [ ! -L "$$hook" ]; then \
	mkdir -p "$$common_dir/hooks" && printf '#!/bin/sh\nmake precommit\n' > "$$hook" && chmod +x "$$hook"; \
	fi

.PHONY: precommit
precommit:
	$(MAKE) precommit-hook
	$(MAKE) fmt
	$(MAKE) test
	$(MAKE) parity
