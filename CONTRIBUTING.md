# Contributing

Before pushing:

- `make precommit` installs a pre-push hook and runs `make fmt`, `make test` and `make parity`.
- `make lint` runs cppcheck.

Code:

- Keep it small. A new op needs a reason in microGPT, a forward, a backward and a gradcheck in `test/test_ops.c`.
- Guard clauses and early returns. Inline single-use helpers. A little duplication is fine.
- One logical statement per line. No line width limit.
- `CHECK` for anything a caller can get wrong. No error handling for impossible states.
- Comments are lowercase, sparse and explain why, not what.
- Fixed-width integer types (`int64_t`). `size_t` only at allocation calls.
