> Based on https://tigerstyle.dev/

C Specific:

- Use explicit types only (e.g., `int64_t`, `float64_t`), no implicit types (`int`, `float`)

Quality:

- Obvious code > clever code
- Comments explain why, not what and are just lower case single lines
- Guard clauses first, early returns, minimize nesting
- Maximize locality: keep related code together, define things near usage
- Minimize lines of code without sacrificing clarity
- Prefer lambdas and inline logic over tiny single-use functions
- Functions should do one coherent thing but not be artificially split
- Functional style: map, filter, reduce, pure functions, immutability
- Procedural when simpler: direct loops, mutation in local scope
- No OOP patterns unless genuinely needed (rare)
- Prefer static/free functions over classes
- Avoid class state and complex initialization
- Data in, data out: functions transform inputs to outputs
- Use structs/records for data, functions for behavior
- Composition over inheritance

Safety:

- Use assert statements in each function to validate inputs and invariants
- Prefer static memory allocation over dynamic memory allocation
- No undefined behavior; rely on explicit code, not compiler optimizations
- Bounded execution: set explicit upper limits on all loops, queues, and recursion
- Fail fast: detect unexpected conditions immediately rather than attempting recovery

---

Building and Testing:

- Run `make fmt` to format code
- Run `make test` to run tests and ensure they pass
- Do NOT run `make run` as it requires downloading large data files
