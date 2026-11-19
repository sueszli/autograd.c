# Design

The engine is the smallest one that trains karpathy's microGPT. `make parity` checks it: 1000 AdamW steps from the reference initial weights must reproduce the reference final weights within 1e-11.

## Tensor

- 2-D, row-major, float64: `data`, `grad`, `rows`, `cols`, `requires_grad`.
- No strides, views, broadcasting or n-D shapes. Every microGPT tensor is a matrix.
- float64 because the reference uses Python floats.

## Tape

- Every op calls `tape_push`, which allocates the output and appends a node: output, up to 4 inputs, a backward function, a scalar `aux` and an owned `ctx` buffer.
- Creation order is a topological order, so `backward(loss)` seeds `loss->grad = 1` and walks the tape in reverse. Nodes whose output got no gradient are skipped.
- The tape owns every op output. `tape_reset()` frees all of them and their `ctx`. Params come from `tensor_new` and are owned by the caller. Call `tape_reset()` after each backward.
- Backward kernels are plain loops that write into `grad_of(input)`. They never call ops, so backward cannot add nodes to the tape it is walking.
- Param grads accumulate across backward calls until the optimizer zeroes them. Calling `backward` twice on one graph is rejected.

## Ops

`src/ops.c`, with forward and backward next to each other:

- `add`, `matmul`, `transpose`, `scale`
- `gather` (embedding lookup, backward scatters and sums duplicates)
- `relu`, `rmsnorm` (no learned weight)
- `causal_mask`, `softmax_rows`
- `slice_cols`, `concat_cols` (split and merge attention heads)
- `masked_cross_entropy` (fused with its softmax)

Attention is built from these in `src/microgpt.c`. It is not a fused op.

## Errors

Plain `assert(cond && "msg")`. CMake passes `-UNDEBUG`, so asserts stay on in Release builds too.

## Parity

- `parity.json` comes from autograd.lean. It holds initial weights, final weights and 1000 tokenized sequences. Linear weights are stored (out, in) and transposed on load.
- Measured margin: rounding-order changes (softmax divide vs multiply, matmul summation order, FMA contraction) stay around 1e-15. Masking one token at step 500 moves the weights by about 1e-1. So 1e-11 separates rounding noise from real differences.
