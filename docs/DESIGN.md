# Autograd Engine Design

Context

- This is a hobby autograd engine in C for learning and experimentation. This is a learning tool, not a hardened library.
- The design prioritizes simplicity and code locality over flexibility. We make explicit tradeoffs: no in-place ops, no retain_graph, no non-leaf gradient retention, no views or aliasing, co-located forward/backward code, arena allocation for graph nodes and dependency counting instead of topological sort. These constraints eliminate entire classes of bugs and keep the codebase small.

Core Structures

- Tensor carries standard fields plus autograd metadata: requires_grad bool, grad pointer (NULL until backward writes to it), grad_fn pointer to the Function that created it (NULL for leaves), and ref_count.
- Function contains: apply function pointer, output pointer to the tensor this function produced, inputs array of parent tensor pointers (fixed-size, maximum 4), num_inputs count, pending_count integer tracking downstream consumers, and ctx void pointer strictly for non-input saved data.

Memory Model

- Tensors use reference counting. Operations return new tensors with ref_count=1; caller owns them. When ref_count hits zero, tensor releases its grad if non-NULL and frees itself.
- Every tensor owns a unique data buffer for its entire lifetime. No aliasing, no shared storage, no views.
- Functions and ctx are allocated from a bump arena owned exclusively by backward(). One active arena per thread. Nested forwards and concurrent graphs within a thread are unsupported.
- The arena is created on first requires_grad operation and stored in thread-local state. backward() frees it. If backward is never called, the arena leaks. Forward without backward is allowed only for debugging or REPL-style exploration; in real use it is misuse. Name it, do not fix it in code.
- Ownership between Tensor and Function is one-way. Tensor->grad_fn points to Function. Function->output points to Tensor. Neither retains the other. Safe because both live in arena and are freed together. Do not add refcounting here.

Invariants

- Non-negotiable. Correctness depends on all of them.
- (1) pending_count equals downstream consumer edges. Incremented once per consumer during graph construction.
- (2) pending_count only incremented for parents with non-NULL grad_fn. Leaves have pending_count zero and never appear in the work queue.
- (3) Backward kernels produce gradients in output shape space. Never pre-reduce. accumulate_grad handles all broadcasting reduction.
- (4) accumulate_grad is the only place gradients are summed. Backward kernels must not call tensor_add on gradients.
- (5) Backward kernels must not read tensor->grad. They receive grad_output as input and write via accumulate_grad. Gradient flow is strictly directional.
- (6) backward() assumes all grad fields are NULL at entry. Only loss->grad is initialized by backward; others are written when reached through the graph.
- (7) Intermediate tensor->grad is allocated during backward and must not be inspected or retained. Only leaf gradients survive.
- (8) backward() is only defined for scalar losses. Calling backward on non-scalar tensors is undefined behavior.
- (9) pending_count must never go negative. Assert in debug builds.
- (10) fn->output->grad must be non-NULL when apply is called. Assert in debug builds. If this fires, dependency counting or accumulation is broken.
- (11) One active arena per thread. No nested forwards.

Restrictions

- No in-place operations on tensors with requires_grad. Fatal error on mutation.
- No retain_graph. Run forward again to differentiate again.
- Only leaf gradients survive backward.
- No views, slicing, or aliasing. Every tensor owns unique data.
- Fixed-size inputs array (maximum 4). Variable-arity ops like concat are out of scope unless the limit is raised. Do not silently truncate.

Graph Construction

- When an operation produces output with requires_grad=true:
    - (1) allocate Function from arena
    - (2) set fn->output
    - (3) populate fn->inputs with requires_grad parents
    - (4) for each parent with non-NULL grad_fn, increment pending_count
    - (5) set output->grad_fn = fn
- ctx is for non-input data only. For add/mul, ctx is NULL. For sum, ctx holds original shape. For matmul, ctx holds transpose flags.

Backward Algorithm

- (1) Assert loss is scalar. Set loss->grad = 1.0.
- (2) Create work queue. Push loss->grad_fn.
- (3) Process queue: for each fn, assert fn->output->grad is non-NULL, call apply, for each parent with non-NULL grad_fn decrement pending_count and assert non-negative and if zero enqueue.
- (4) Free arena.
- Queue order does not affect correctness. Do not rely on FIFO or LIFO semantics.

Gradient Accumulation

- accumulate_grad(tensor, new_grad) takes ownership of new_grad. Reduces to tensor's shape. If tensor->grad is NULL, assigns directly. Otherwise sums, releases old, assigns result.
- Single location for gradient addition and broadcast handling.

Operation Implementation

- Forward and backward adjacent in same file.
- mul: Forward computes result. If requires_grad, allocate Function, set output, populate inputs, increment pending_counts. ctx is NULL. Backward reads inputs, computes grad_a = grad_output * b and grad_b = grad_output * a, calls accumulate_grad.
- sum: Forward computes scalar. ctx holds original shape. Backward broadcasts grad_output to shape, calls accumulate_grad.
- Apply signature: void apply(Function *self, void *ctx, Tensor *grad_output).

API Surface

- tensor_create, tensor_retain, tensor_release, tensor_zero_grad.
- tensor_add, tensor_mul, tensor_matmul, tensor_sum.
- backward(loss).

File Organization

- src/tensor.c: Tensor, refcounting, creation, zero_grad.
- src/autograd.c: Function, arena, backward loop, accumulate_grad. Nothing else.
- src/ops/add.c, mul.c, matmul.c, reduce.c: each op with forward and backward adjacent.

Development Phases

- Phase 1: Tensor with refcounting, arena, Function, backward loop without apply. Assert pending_count >= 0.
- Phase 2: accumulate_grad, add op, finite difference verification.
- Phase 3: mul, sum, matmul, each verified.
- Phase 4: Diamond DAG (z = x + x), valgrind, edge cases.
