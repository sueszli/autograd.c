#include "autograd.h"
#include "tensor.h"
#include "utils/aligned_alloc.h"
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//
// arena allocator
//

#define ARENA_CAPACITY (4 * 1024 * 1024) // 4MB

static __thread Arena *thread_arena = NULL;

static Arena *arena_create(void) {
    Arena *arena = (Arena *)malloc(sizeof(Arena));
    assert(arena != NULL && "malloc failed");
    arena->memory = safe_aligned_alloc(ARENA_CAPACITY);
    arena->capacity = ARENA_CAPACITY;
    arena->offset = 0;
    return arena;
}

static Arena *get_or_create_arena(void) {
    if (thread_arena == NULL) {
        thread_arena = arena_create();
    }
    return thread_arena;
}

Function *arena_alloc_function(void) {
    Arena *arena = get_or_create_arena();

    // align to 8 bytes
    size_t aligned_offset = (arena->offset + 7) & ~(size_t)7;

    // oom, but in prod code you would allocate a new arena
    if (aligned_offset + sizeof(Function) > arena->capacity) {
        assert(false && "arena out of memory");
    }

    Function *fn = (Function *)((char *)arena->memory + aligned_offset);
    arena->offset = aligned_offset + sizeof(Function);

    memset(fn, 0, sizeof(Function));

    return fn;
}

void arena_free(void) {
    if (thread_arena) {
        free(thread_arena->memory);
        free(thread_arena);
        thread_arena = NULL;
    }
}

//
// backward pass
//

#define MAX_QUEUE_SIZE 10000

typedef struct {
    Function *items[MAX_QUEUE_SIZE];
    int front;
    int rear;
    int count;
} Queue;

static void queue_init(Queue *q) {
    q->front = 0;
    q->rear = 0;
    q->count = 0;
}

static void queue_push(Queue *q, Function *fn) {
    assert(q->count < MAX_QUEUE_SIZE && "queue overflow");
    q->items[q->rear] = fn;
    q->rear = (q->rear + 1) % MAX_QUEUE_SIZE;
    q->count++;
}

static Function *queue_pop(Queue *q) {
    assert(q->count > 0 && "queue underflow");
    Function *fn = q->items[q->front];
    q->front = (q->front + 1) % MAX_QUEUE_SIZE;
    q->count--;
    return fn;
}

static bool queue_empty(const Queue *q) { return q->count == 0; }

void backward(Tensor *loss) {
    assert(loss != NULL);
    assert(loss->ndim == 0 && "loss must be scalar");
    assert(loss->size == 1 && "loss must be scalar");

    // initialize loss->grad to 1.0 (d loss / d loss = 1)
    const uint64_t shape_scalar[] = {};
    float32_t one = 1.0f;
    loss->grad = tensor_create(&one, shape_scalar, 0, false);

    // if loss has no grad_fn, it's a leaf and there's nothing to backprop
    if (loss->grad_fn == NULL) {
        arena_free();
        return;
    }

    // work queue
    Queue queue;
    queue_init(&queue);
    queue_push(&queue, loss->grad_fn);

    // process queue
    while (!queue_empty(&queue)) {
        Function *fn = queue_pop(&queue);
        assert(fn->output != NULL);
        assert(fn->output->grad != NULL && "fn->output->grad is NULL");

        // call backward kernel
        if (fn->apply != NULL) {
            fn->apply(fn, fn->output->grad);
        }

        // for each parent with non-NULL grad_fn, decrement pending_count
        for (uint32_t i = 0; i < fn->num_inputs; i++) {
            Tensor *parent = fn->inputs[i];
            if (parent == NULL) {
                continue;
            }

            // only tensors with grad_fn have pending_count
            if (parent->grad_fn != NULL) {
                assert(parent->grad_fn->pending_count > 0 && "pending_count already zero");
                parent->grad_fn->pending_count--;

                // if pending_count reaches zero, all consumers processed
                if (parent->grad_fn->pending_count == 0) {
                    queue_push(&queue, parent->grad_fn);
                }
            }
        }
    }

    arena_free();
}

//
// gradient accumulation
//

// helper function to check if two shapes are broadcastable
static bool shapes_equal(const Tensor *a, const Tensor *b) {
    if (a->ndim != b->ndim) {
        return false;
    }
    for (uint64_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }
    return true;
}

void accumulate_grad(Tensor *tensor, Tensor *new_grad) {
    assert(tensor != NULL);
    assert(new_grad != NULL);
    if (!shapes_equal(tensor, new_grad)) {
        assert(false && "shape mismatch in accumulate_grad. broadcast reduction not yet implemented");
    }

    // if tensor->grad is NULL, assign directly
    if (tensor->grad == NULL) {
        tensor->grad = new_grad;
        return;
    }

    Tensor *summed = tensor_zeros(tensor->shape, tensor->ndim, false);
    for (uint64_t i = 0; i < tensor->size; i++) {
        summed->data[i] = tensor->grad->data[i] + new_grad->data[i];
    }

    tensor_free(tensor->grad);
    tensor_free(new_grad);

    tensor->grad = summed;
}
