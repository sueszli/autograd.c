# TODO

Goal: the smallest autograd engine that trains karpathy's microGPT, verified by `make parity` like autograd.lean: train 1000 AdamW steps from `parity.json` and match the reference final weights within 1e-11.

## Roadmap

One small PR per step.

- [x] 1. Delete the CNN path (conv, pooling, batchnorm, dropout, layers, CIFAR loader, augment, sigmoid/tanh/gelu, mse/bce, SGD) and fix the build: project name, libm, `-ffp-contract=off`, `-Werror`, Linux-compatible Makefile.
- [x] 2. Core engine: 2-D float64 tensor, tape-based backward, always-on `CHECK`. Ops `add`, `matmul`, `transpose`, `gather`, plus a finite-difference gradient check. Backward kernels are plain loops that never call ops, so no grad-mode flag is needed.
- [x] 3. Ops `scale`, `relu`, `rmsnorm`, `causal_mask`, `softmax_rows`, with gradchecks and composed-graph regression tests (MLP, softmax of matmul, one causal attention head).
- [x] 4. Ops `slice_cols`, `concat_cols`, `masked_cross_entropy`, with gradchecks. Cross-entropy shares its row softmax with `softmax_rows`.
- [x] 5. MicroGPT forward and AdamW, with unit tests: uniform loss at zero weights, a gradcheck over all 4240 weights, the first AdamW step, and a short training run.
- [ ] 6. `make parity`: load `parity.json` with cJSON, train 1000 steps, compare final weights within 1e-11.
- [ ] 7. Inference sampling, README, rewrite `docs/design.md` (it still describes the old engine), pre-push hook.
