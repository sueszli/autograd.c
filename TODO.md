# TODO

Goal: the smallest autograd engine that trains karpathy's microGPT, verified by `make parity` like autograd.lean: train 1000 AdamW steps from `parity.json` and match the reference final weights within 1e-11.

## Roadmap

One small PR per step.

- [x] 1. Delete the CNN path (conv, pooling, batchnorm, dropout, layers, CIFAR loader, augment, sigmoid/tanh/gelu, mse/bce, SGD) and fix the build: project name, libm, `-ffp-contract=off`, `-Werror`, Linux-compatible Makefile.
- [x] 2. Core engine: 2-D float64 tensor, tape-based backward, always-on `CHECK`. Ops `add`, `matmul`, `transpose`, `gather`, plus a finite-difference gradient check. Backward kernels are plain loops that never call ops, so no grad-mode flag is needed.
- [ ] 3. Ops `scale`, `relu`, `rmsnorm`, `causal_mask`, `softmax_rows`, with gradchecks and composed-graph regression tests.
- [ ] 4. Ops `slice_cols`, `concat_cols`, `masked_cross_entropy`, with gradchecks.
- [ ] 5. MicroGPT forward and AdamW, with unit tests.
- [ ] 6. `make parity`: load `parity.json` with cJSON, train 1000 steps, compare final weights within 1e-11.
- [ ] 7. Inference sampling, README, rewrite `docs/design.md` (it still describes the old engine), pre-push hook.
