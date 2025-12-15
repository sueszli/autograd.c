# Understanding the Backward Pass

## The Problem

When computing gradients, we need to process operations in **reverse topological order** - we can only compute gradients for a tensor after **all** of its downstream consumers have finished computing their gradients.

## Key Concept: `pending_count`

`pending_count` tracks **how many downstream operations haven't finished yet**. Think of it as:
- "How many children still need to give me their gradients?"

When `pending_count` reaches 0, it means all children have finished, so we can safely process this node.

## Example: Why We Need `pending_count`

Consider this computation graph:

```
a (leaf)
  \
   \---> mul1 ---> c
  /              /
b (leaf)        /
  \            /
   \---> mul2 -/
```

Forward pass:
- `c = mul1(a, b) + mul2(a, b)`  (simplified: `c` uses `a` twice)
- `loss = sum(c)`

During forward pass:
- `mul1` creates `Function` with `inputs[0] = a, inputs[1] = b`
- `mul2` creates `Function` with `inputs[0] = a, inputs[1] = b`
- Both increment `a->grad_fn->pending_count++` (if `a` has a `grad_fn`)
- So `a->grad_fn->pending_count = 2` (two children)

## Backward Pass Walkthrough

### Step 1: Initialize
```
loss.grad = 1.0
queue = [loss->grad_fn]  // the sum operation
```

### Step 2: Process `loss->grad_fn` (sum)
```
fn = pop(queue)  // sum function
fn->apply(fn, loss->grad)  // computes grad for c
  → calls accumulate_grad(c, grad_c)

// Now check parents of sum: c
for parent in [c]:
    if c->grad_fn:  // yes, c has grad_fn (it was created by add)
        c->grad_fn->pending_count--  // was 0, now -1? Wait...
```

Wait, let's trace this more carefully. The actual graph is:

```
a --[mul1]--> c1
b --[mul1]--> c1
a --[mul2]--> c2  
b --[mul2]--> c2
c1 + c2 = c
sum(c) = loss
```

So:
- `c1->grad_fn` is `mul1`, `pending_count = 0` initially
- `c2->grad_fn` is `mul2`, `pending_count = 0` initially
- `c->grad_fn` is `add`, `pending_count = 0` initially
- `loss->grad_fn` is `sum`, `pending_count = 0` initially

But `a` might be used in multiple places. Let's say `a` is created by some op `op_a`:
- `a->grad_fn = op_a`, `pending_count = 2` (used by mul1 and mul2)

### Step 3: Process `add` function
```
fn = pop(queue)  // add function
fn->apply(fn, c->grad)  // computes grads for c1 and c2
  → accumulate_grad(c1, grad_c1)
  → accumulate_grad(c2, grad_c2)

// Check parents: c1, c2
for parent in [c1, c2]:
    if c1->grad_fn:  // mul1
        mul1->pending_count--  // 0 → -1? No wait, it starts at 0
        if pending_count == 0:  // yes, it's 0 now
            push(queue, mul1)
    
    if c2->grad_fn:  // mul2
        mul2->pending_count--  // 0 → -1? No...
```

Hmm, I see the confusion. Let me check the actual initialization...

Looking at the code:
- When `mul1` is created, `mul1->pending_count = 0`
- Then: `if (a->grad_fn) a->grad_fn->pending_count++`
- So `a->grad_fn->pending_count` is incremented, not `mul1->pending_count`

Ah! The `pending_count` belongs to the **parent tensor's grad_fn**, not the current operation's grad_fn!

## Correct Understanding

When operation `op` creates output `result`:
- `result->grad_fn = op` (the operation that created it)
- `op->pending_count = 0` (starts at 0)
- For each input `parent`:
  - If `parent->grad_fn` exists (parent was created by some operation):
    - `parent->grad_fn->pending_count++` (increment the PARENT's operation's pending count)

So `pending_count` on `parent->grad_fn` means: "how many of my children haven't finished backward pass yet?"

## Corrected Example

```
a = some_op(x)  // a->grad_fn = some_op, some_op->pending_count = 0
b = leaf

c1 = mul(a, b)  // c1->grad_fn = mul1, mul1->pending_count = 0
                // some_op->pending_count++  (now = 1)

c2 = mul(a, b)  // c2->grad_fn = mul2, mul2->pending_count = 0  
                // some_op->pending_count++  (now = 2)

c = add(c1, c2) // c->grad_fn = add_op, add_op->pending_count = 0
                // mul1->pending_count++  (now = 1)
                // mul2->pending_count++  (now = 1)

loss = sum(c)   // loss->grad_fn = sum_op, sum_op->pending_count = 0
                // add_op->pending_count++  (now = 1)
```

## Backward Pass

```
loss.grad = 1.0
queue = [sum_op]

// Iteration 1: Process sum_op
fn = sum_op
sum_op->apply(sum_op, loss->grad)  // computes grad for c
  → accumulate_grad(c, grad_c)

// Check parents of sum_op: c
if c->grad_fn (add_op):
    add_op->pending_count--  // 1 → 0
    if pending_count == 0:
        push(queue, add_op)  // ✅ Safe to process now

// Iteration 2: Process add_op
fn = add_op
add_op->apply(add_op, c->grad)  // computes grads for c1 and c2
  → accumulate_grad(c1, grad_c1)
  → accumulate_grad(c2, grad_c2)

// Check parents of add_op: c1, c2
if c1->grad_fn (mul1):
    mul1->pending_count--  // 1 → 0
    if pending_count == 0:
        push(queue, mul1)  // ✅ Safe to process now

if c2->grad_fn (mul2):
    mul2->pending_count--  // 1 → 0
    if pending_count == 0:
        push(queue, mul2)  // ✅ Safe to process now

// Iteration 3: Process mul1
fn = mul1
mul1->apply(mul1, c1->grad)  // computes grads for a and b
  → accumulate_grad(a, grad_a_from_mul1)
  → accumulate_grad(b, grad_b_from_mul1)

// Check parents of mul1: a, b
if a->grad_fn (some_op):
    some_op->pending_count--  // 2 → 1
    // pending_count != 0, so DON'T push yet (mul2 still needs to finish)

if b->grad_fn:  // b is a leaf, no grad_fn
    // skip

// Iteration 4: Process mul2
fn = mul2
mul2->apply(mul2, c2->grad)  // computes grads for a and b
  → accumulate_grad(a, grad_a_from_mul2)  // accumulates with existing grad_a_from_mul1
  → accumulate_grad(b, grad_b_from_mul2)  // accumulates with existing grad_b_from_mul1

// Check parents of mul2: a, b
if a->grad_fn (some_op):
    some_op->pending_count--  // 1 → 0
    if pending_count == 0:
        push(queue, some_op)  // ✅ Safe to process now (both mul1 and mul2 done)

if b->grad_fn:  // b is a leaf, no grad_fn
    // skip

// Iteration 5: Process some_op
fn = some_op
some_op->apply(some_op, a->grad)  // computes grad for x
  → accumulate_grad(x, grad_x)

// Check parents of some_op: x
if x->grad_fn:  // depends on what created x
    x->grad_fn->pending_count--
    if pending_count == 0:
        push(queue, x->grad_fn)

// Queue empty, done!
```

## Why `accumulate_grad`?

When a tensor is used in multiple operations (like `a` in `mul1` and `mul2`), it receives gradients from multiple sources. We need to **sum** these gradients:

```
a->grad = grad_from_mul1 + grad_from_mul2
```

This is what `accumulate_grad` does - it adds the new gradient to any existing gradient.

## Simple Concrete Example: `z = add(x, x)`

From `test_add_backward_diamond`:
```c
Tensor *x = tensor_create(&x_data, shape, 0, true);  // x is a leaf
Tensor *z = tensor_add(x, x);  // z = x + x
backward(z);
```

### Forward Pass Setup

```
x (leaf, requires_grad=true, grad_fn=NULL)
  \
   \--[add_op]--> z
  /
x (same tensor, used twice)
```

When `tensor_add(x, x)` executes:
1. Creates `add_op` Function:
   - `add_op->output = z`
   - `add_op->inputs[0] = x`
   - `add_op->inputs[1] = x`  // same tensor!
   - `add_op->pending_count = 0`
   - `z->grad_fn = add_op`

2. Since `x` is a leaf (no `grad_fn`), no `pending_count` increments happen

### Backward Pass

```
Step 1: Initialize
  z->grad = 1.0  // d(z)/d(z) = 1
  queue = [add_op]

Step 2: Process add_op
  fn = add_op
  add_op->apply(add_op, z->grad)  // computes gradients for inputs
    → For input[0] (x): grad_x1 = 1.0, accumulate_grad(x, grad_x1)
    → For input[1] (x): grad_x2 = 1.0, accumulate_grad(x, grad_x2)
      // accumulate_grad adds: x->grad = 1.0 + 1.0 = 2.0 ✅

  // Check parents: x (appears twice, but it's the same tensor)
  if x->grad_fn:  // NULL, x is a leaf
    // skip

  // Queue empty, done!
```

Result: `x->grad = 2.0` ✅ (correct: d(x+x)/dx = 2)

## More Complex Example: `w = add(mul(x,y), x)`

From `test_mul_add_chain`:
```c
Tensor *x = tensor_create(&x_data, shape, 0, true);
Tensor *y = tensor_create(&y_data, shape, 0, true);
Tensor *z = tensor_mul(x, y);  // z = x * y
Tensor *w = tensor_add(z, x);  // w = z + x = (x*y) + x
backward(w);
```

### Forward Pass Setup

```
x (leaf)
  \
   \--[mul_op]--> z
  /              /
y (leaf)        /
  \            /
   \--[add_op]--> w
```

When `tensor_mul(x, y)` executes:
- `mul_op->pending_count = 0`
- `z->grad_fn = mul_op`

When `tensor_add(z, x)` executes:
- `add_op->pending_count = 0`
- `w->grad_fn = add_op`
- Since `z` has `grad_fn` (mul_op), increment: `mul_op->pending_count++` (now = 1)
- Since `x` is a leaf, no increment

### Backward Pass

```
Step 1: Initialize
  w->grad = 1.0
  queue = [add_op]

Step 2: Process add_op
  fn = add_op
  add_op->apply(add_op, w->grad)  // w = z + x
    → For z: grad_z = 1.0, accumulate_grad(z, grad_z)
    → For x: grad_x_from_add = 1.0, accumulate_grad(x, grad_x_from_add)

  // Check parents: z, x
  if z->grad_fn (mul_op):
    mul_op->pending_count--  // 1 → 0
    if pending_count == 0:
      push(queue, mul_op)  // ✅ Safe to process

  if x->grad_fn:  // NULL, skip

Step 3: Process mul_op
  fn = mul_op
  mul_op->apply(mul_op, z->grad)  // z = x * y, so dz/dx = y, dz/dy = x
    → For x: grad_x_from_mul = y->data[0] = 3.0, accumulate_grad(x, grad_x_from_mul)
      // x->grad = 1.0 + 3.0 = 4.0 ✅
    → For y: grad_y = x->data[0] = 2.0, accumulate_grad(y, grad_y)
      // y->grad = 2.0 ✅

  // Check parents: x, y
  if x->grad_fn:  // NULL, skip
  if y->grad_fn:  // NULL, skip

  // Queue empty, done!
```

Result: `x->grad = 4.0` ✅ (d((x*y)+x)/dx = y + 1 = 3 + 1 = 4)
        `y->grad = 2.0` ✅ (d((x*y)+x)/dy = x = 2)

## Summary

1. **`pending_count`** tracks how many downstream operations haven't finished backward pass yet
2. When an operation finishes, it decrements `pending_count` on all its input operations
3. When `pending_count` reaches 0, that operation is safe to process (all children done)
4. **`accumulate_grad`** sums gradients when a tensor receives contributions from multiple children
5. The queue ensures we process operations in the correct order (topological sort)
