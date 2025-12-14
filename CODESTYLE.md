Code Style:

- Guard clauses first, early returns, minimize nesting
- Maximize locality: keep related code together, define things near usage
- Minimize lines of code without sacrificing clarity
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
