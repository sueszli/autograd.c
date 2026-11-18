# TODO

Goal: the smallest autograd engine that trains karpathy's microGPT, verified by `make parity` like autograd.lean: train 1000 AdamW steps from `parity.json` and match the reference final weights within 1e-11.

## Roadmap

One small PR per step.

- [x] 1. Delete the CNN path (conv, pooling, batchnorm, dropout, layers, CIFAR loader, augment, sigmoid/tanh/gelu, mse/bce, SGD) and fix the build: project name, libm, `-ffp-contract=off`, `-Werror`, Linux-compatible Makefile.
- [ ] 2. Core engine: 2-D float64 tensor, tape-based backward, grad mode off inside `backward()`, always-on `CHECK`. Ops `add`, `matmul`, `transpose`, `gather`, plus a finite-difference gradient check.
- [ ] 3. Ops `scale`, `relu`, `rmsnorm`, `causal_mask`, `softmax_rows`, with gradchecks and composed-graph regression tests.
- [ ] 4. Ops `slice_cols`, `concat_cols`, `masked_cross_entropy`, with gradchecks.
- [ ] 5. MicroGPT forward and AdamW, with unit tests.
- [ ] 6. `make parity`: load `parity.json` with cJSON, train 1000 steps, compare final weights within 1e-11.
- [ ] 7. Inference sampling, README, `docs/design.md`, pre-push hook.

## Known issues in code that step 2 replaces

These come from the 2026-11-18 review of `src/tensor.c` and `src/autograd.c`.

- `Function->inputs` does not retain tensors, and `tensor_retain`/`tensor_release` are unused.
- `arena_free()` leaves dangling `grad_fn` pointers on live tensors.
- `backward()` overwrites `loss->grad` without freeing it.
- `backward()` does not disable graph building, so kernels that call ops bump `pending_count` and drop gradients.
- Asserts are the only error handling and vanish under `-DNDEBUG`.
- `strides` are stored but every tensor is contiguous. Each tensor does 3 heap allocations.
- `docs/design.md` describes behavior the code does not have.
