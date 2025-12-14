C Specific:

- Use explicit types only (e.g., `int64_t`, `float64_t`), no implicit types (`int`, `float`)
- Avoid architecture-dependent types (e.g., `usize`, `long`) to ensure portability

Quality:

- Obvious code > clever code
- Maximize locality: Maximize locality: keep related code together, define things near usage, minimize variable scope. Centralize flow control. Branching in parents, pure logic in leaves.
- Guard clauses first, early returns, minimize nesting
- Minimize lines of code without sacrificing clarity
- Functions should do one coherent thing (ideally <70 lines) but not be artificially split
- Prefer lambdas and inline logic over tiny single-use functions
- Use variables / lambdas with clear names where it makes sense instead of using comments for long conditions or unclear segments
- Comments explain why, not what. are lowercase. maintain context on decisions
- Variable names include units/qualifiers (e.g., timeout_ms, size_bytes)
- Distinguish types: Index (0-based), Count (1-based), and Size (memory bytes)
- Functional style: map, filter, reduce, pure functions, immutability
- Procedural when simpler: direct loops, mutation in local scope
- No OOP patterns unless genuinely needed (rare)
- Data in, data out: functions transform inputs to outputs
- Composition over inheritance

Safety:

- Zero technical debt policy: fix issues immediately, don't rely on future refactoring
- Assert inputs, outputs, and invariants in every function
- Pair assertions: check critical data at multiple points to catch inconsistencies
- No undefined behavior: rely on explicit code, not compiler optimizations
- Bounded execution: fixed upper limits on all loops, queues, and recursion
- Fail fast: detect unexpected conditions immediately; crash rather than corrupt
- Treat all compiler warnings as errors
- Strongly prefer static memory allocation: allocate all memory at startup

Pre-commit:

- Run `make fmt` to format code
- Run `make test` to run tests and ensure they pass
- Do NOT run `make run` as it requires downloading large data files
