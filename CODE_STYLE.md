C Specific:

- Use explicit types only (e.g., `int64_t`, `float64_t`), no implicit types (`int`, `float`)
- Avoid architecture-dependent types (e.g., `size_t`, `long`) to ensure portability. use fixed-size explicit types instead

Quality:

- Obvious code > clever code
- Maximize locality: Keep related code together, define things near usage, minimize variable scope
- Centralize control flow: Branching logic belongs in parents. leaf functions should be pure logic
- Guard clauses first, early returns, minimize nesting
- Minimize lines of code without sacrificing clarity
- Functions should do one coherent thing (ideally <70 lines) but not be artificially split
- Prefer lambdas and inline logic over tiny single-use functions
- Use named variables/lambdas to simplify complex conditionals (decompose conditionals)
- Comments explain *why*, not *what*. use lowercase single lines and ASCII illustrations.
- Naming:
      - Variable names must include units/qualifiers (e.g., `timeout_ms`, `size_bytes`)
      - Distinguish types: Index (0-based), Count (1-based), and Size (memory bytes)
- Functional Style:
      - Prefer pure functions (data in, data out) and immutability
      - Use map/filter/reduce patterns where clearer
- Procedural Style: Use direct loops and local mutation when simpler or more performant
- Use structs/records for data and functions for behavior (no OOP/inheritance)

Safety:

- Zero technical debt: Fix issues immediately. never rely on future refactoring
- Assert aggressively: Validate inputs, outputs and invariants in every function
- Pair assertions: Check critical data at multiple points/methods to catch internal inconsistencies
- Bounded execution: Set fixed upper limits on all loops, queues and recursion depth
- Fail fast: Detect unexpected conditions immediately. crash rather than corrupt state
- No undefined behavior: rely on explicit code, not compiler optimizations
- Treat all compiler warnings as errors
- Controlled Memory:
    - Strongly prefer static allocation over dynamic allocation.
    - When dynamic allocation is necessary (e.g., graph construction), use arenas or memory pools.
    - Enforce strict upper bounds on memory usage to prevent leaks/OOM.
    - Assert Allocation: Wrap every arena allocation, malloc, or resource claim in an assert to guarantee success (e.g., `assert(ptr != NULL)`).

Pre-commit:

- Add unit tests for each new function and feature.
- Run `make fmt` to format code
- Run `make test` to run tests and ensure they pass
- Do NOT run `make run` as it requires downloading large data files
