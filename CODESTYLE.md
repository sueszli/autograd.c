C Style:

- Use explicit types only (e.g., `int64_t`, `float64_t`), no implicit types (`int`, `float`)

Code Style:

- Guard clauses first, early returns, minimize nesting
- Use assert statements in each function to validate inputs and verify outputs where it makes sense
- Maximize locality: keep related code together, define things near usage
- Minimize lines of code without sacrificing clarity

Building and Testing:

- Run `make fmt` to format code
- Run `make test` to run tests and ensure they pass
- Do NOT run `make run` as it requires downloading large data files
- Prefer lambdas and inline logic over tiny single-use functions
- Functions should do one coherent thing but not be artificially split

Architecture:

- Prefer static/free functions over classes
- Avoid class state and complex initialization
- Data in, data out: functions transform inputs to outputs
- Use structs/records for data, functions for behavior
- Composition over inheritance

Paradigm:

- Functional style: map, filter, reduce, pure functions, immutability
- Procedural when simpler: direct loops, mutation in local scope
- No OOP patterns unless genuinely needed (rare)

Readability:

- Obvious code > clever code
- Name things by what they represent, not how they work
- Comments explain why, not what and are just lower case single lines
