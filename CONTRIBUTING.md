# (1) Safety

- Zero Technical Debt: Fix issues immediately. Never rely on future refactoring.
- Testing:
      - Mandatory Unit Tests: Every new feature must be accompanied by unit tests.
      - Coverage: The more tests, the better. Cover edge cases, boundary conditions and failure modes.
- Assert Aggressively: Validate inputs, outputs, tensor shapes and invariants in every function.
- Pair Assertions: Check critical data at multiple points to catch internal inconsistencies.
- Bounded Execution: Set fixed upper limits on all loops, queues, and recursion depth (especially for graph traversal).
- Fail Fast: Detect unexpected conditions immediately. Crash rather than corrupt state.
- No Undefined Behavior: Rely on explicit code, not compiler optimizations. Treat all compiler warnings as errors.
- Controlled Memory:
    - Strongly prefer static allocation over dynamic allocation.
    - When dynamic allocation is necessary (e.g., graph construction), use arenas or memory pools*
    - Enforce strict upper bounds on memory usage.
    - Assert Allocation: Wrap every arena allocation, `malloc`, or resource claim in an assert (e.g., `assert(ptr != NULL)`).

C Specific:

- Use explicit types only (e.g., `int64_t`, `float64_t`), no implicit types (`int`, `float`)
- Avoid architecture-dependent types (e.g., `size_t`, `long`) to ensure portability; use fixed-size explicit types instead
- Strict Const Correctness: Pointers are `const` by default. Only drop `const` when mutation is strictly required.
- Explicit Mutation: Avoid manipulating function arguments or causing side effects
    - Exception: If copying has an extremely high memory cost (e.g., large tensors), mutation is allowed but must be obvious in the naming (e.g., `out_result`, `tensor_mut`).

Pre-commit

- Add unit tests for each new function and feature
- Run `make fmt`, `make lint`, `make test`
- Do NOT run `make run` as it requires downloading large data files

# (2) Quality

- Obvious Code > Clever Code
- Maximize Locality: Keep related code together. Define things near usage. Minimize variable scope.
- Centralize Control Flow: Branching logic belongs in parents. leaf functions should be pure logic.
- Guard Clauses: Handle checks first, return early, minimize nesting.
- Functions: Do one coherent thing (ideally <70 lines). Prefer lambdas/inline logic over tiny single-use functions.
- Decompose Conditionals: Use named variables to simplify complex `if` conditions.
- Naming & Comments:
    - Comments explain *why*, not *what*; use lowercase single lines. ASCII illustrations are welcome.
- Paradigm Balance:
    - Functional: Prefer pure functions (data in, data out) and immutability for logic.
    - Procedural: Use direct loops and local mutation when simpler or significantly more performant.

# (3) Performance

- Follow Data-Oriented Design (DoD) principles
- Design for Hardware: Organize data to match how the hardware reads it (cache lines).
- Struct of Arrays (SoA): Prefer SoA over Array of Structs (AoS) for heavy computation to maximize SIMD usage.
- Data Alignment: Ensure critical data (tensors) is **aligned** (e.g., 64 bytes) for SIMD efficiency. Assert alignment on access.
- Batch Processing: Write functions that transform arrays of data rather than single elements (data transformation > object interaction).
- Existence-based Processing: Filter data *before* processing so loops run on contiguous, valid data (avoid `if (obj->active)` inside hot loops).

---

> Based on https://tigerstyle.dev/
