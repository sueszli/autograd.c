# TODO

Ordered by priority. Items marked "confirmed" were reproduced with ASan/UBSan builds.

## Gradient correctness

- [ ] Backward kernels build graph nodes during `backward()` (confirmed). `matmul_backward` calls `tensor_transpose(a)`, `div_backward` calls `tensor_mul(b, b)` and `tensor_div` without disabling grad. This bumps `pending_count` on a parent that is mid-backward, so it never reaches zero and the whole upstream graph is skipped.
  - 2-layer MLP `sum(relu(x@W1)@W2)`: `W1.grad` is NULL.
  - `sum(x/(x*x))` at `x=[1,2]` gives `[+1, +0.25]`, expected `[-1, -0.25]`.
  - In `main.c` the `fc2` matmul transposes `fc1_relu`, so conv1-3 and fc1 likely never train. This likely explains the loss stuck near 2.29.
  - Fix: a global no-grad flag that `backward()` sets, checked by one shared graph-node helper.
- [ ] `tensor_softmax` is composed from `max`, `sub`, `sum` and `div`, each of which records a node and bumps the input's `pending_count`, then frees those intermediates and calls `expf` in place on `shifted` (confirmed: `w.grad` is NULL for `softmax(x*w)`). Implement it as a single fused op.
- [ ] `linear_forward` and `sequential_forward` free tensors that the graph still references (confirmed: heap-use-after-free at `src/ops/arithmetic_backward.c:77`).
- [ ] Copy-instead-of-op paths cut the graph: dropout in eval mode or with `p<=0`, empty `Sequential`, `transpose` on `ndim<2`. They return a copy with `requires_grad=true` and no `grad_fn` (confirmed for dropout in eval mode).
- [ ] GELU forward uses exact `erf`, backward uses the tanh-approximation derivative. Make them match.
- [ ] `max_backward` gives the full gradient to every tied max element.
- [ ] `tensor_softmax_backward(t)` computes only the Jacobian diagonal. Its tests in `test/test_activations_backward.c` assert the wrong result. Remove it or fix it together with the tests.

## Memory and ownership

- [ ] `Function->inputs` does not retain tensors. `tensor_retain`/`tensor_release` are unused outside `tensor.c` (0 calls, 88 calls to `tensor_free`). Retain inputs when recording a node and release them when the graph is freed.
- [ ] `forward_cnn` in `main.c` never frees its intermediates. Every batch leaks all activations.
- [ ] `ctx` is `malloc`ed, not arena-allocated, and freed only inside `apply`. Nodes that never run leak it (every eval forward, every node hit by the bug above).
- [ ] `arena_free()` leaves dangling `grad_fn` pointers on live tensors.
- [ ] `backward()` overwrites `loss->grad` without freeing it.
- [ ] Add a global no-grad mode and use it in `evaluate()`. Remove the `disable_grad` macro overload on `tensor_mul`.

## Build and release

- [ ] Asserts are the only error handling, and CMake Release adds `-DNDEBUG`. In `run-release`, malloc failure, shape mismatch and arena overflow continue silently. Arena overflow becomes an out-of-bounds write. Use an always-on check macro for runtime errors.
- [ ] `-O2` in `add_compile_options` overrides Release's `-O3`.
- [ ] `cifar10.c` loads data in an `__attribute__((constructor))`. Every binary linking the library, including all tests, aborts if the dataset is missing. Load lazily from `main`.
- [ ] `M_PI` is not standard C. The build fails on Linux with `-std=c2x` (`src/ops/activations_backward.c:139`).
- [ ] Makefile is macOS-only (`sysctl -n hw.ncpu`, `/opt/homebrew` clang, `codesign`, `leaks`).
- [ ] CMake project is named `defer`.

## Performance

- [ ] Broadcast binary ops do a div/mod index conversion per dimension and a function-pointer call per element. Add a same-shape fast path.
- [ ] matmul is naive i-j-k with a strided inner loop over `b`. Use i-k-j order or blocking. Backward materializes full transposes.
- [ ] conv2d is a 7-deep loop with input channels innermost and copies the padded input every call. Consider im2col + matmul.
- [ ] Strides are stored but views are not supported, so every tensor is contiguous. Either add views or drop strides.
- [ ] Each tensor does 3 heap allocations. Store `shape`/`strides` inline as `[MAX_NDIM]` arrays.

## Code quality

- [ ] The ~15-line Function setup block is copy-pasted about 25 times. Replace it with one helper that also checks the no-grad flag.
- [ ] `docs/design.md` is out of date: it says `accumulate_grad` handles broadcast reduction (the code asserts on mismatch and kernels call `unbroadcast`) and that `ctx` lives in the arena (it is `malloc`ed).
- [ ] Unused duplicate helpers (`tensor_sigmoid_backward`, `tensor_relu_backward`, etc.) are exported and tested but not used by the engine.
- [ ] Cross-entropy's `sum_exp < 1.0f` clamp can never fire. Labels are stored as floats.

## Tests

- [ ] Add a finite-difference gradient check.
- [ ] Add composed-graph tests where intermediate (non-leaf) tensors feed matmul, div and softmax: `matmul->relu->matmul`, `x/(x*x)`, `softmax(x*w)`. Current autograd tests only feed leaves into a single op.
- [ ] Layer tests only check the forward pass. Add backward checks.
