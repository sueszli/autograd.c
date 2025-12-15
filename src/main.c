#include "autograd.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("Running Autograd Experiment...\n");

    // Create scalar tensors x and y
    // x = 2.0
    uint64_t shape[] = {1};
    Tensor *x = tensor_create(NULL, shape, 0, true); // Scalar
    x->data[0] = 2.0f;

    // y = 3.0
    Tensor *y = tensor_create(NULL, shape, 0, true); // Scalar
    y->data[0] = 3.0f;

    printf("Created tensors x = %f, y = %f\n", x->data[0], y->data[0]);

    // z = (x + y) * x
    //   = x^2 + y*x
    // dz/dx = 2x + y = 2(2) + 3 = 7
    // dz/dy = x = 2

    Tensor *sum = tensor_add(x, y);
    Tensor *z = tensor_mul(sum, x);

    printf("Computed z = (x + y) * x\n");
    tensor_print(z);

    // Backward pass
    printf("Running backward()...\n");
    backward(z, NULL); // implicit grad = 1.0

    // Print gradients
    if (x->grad) {
        printf("x->grad = %f (Expected: 7.0)\n", x->grad->data[0]);
    } else {
        printf("x->grad is NULL!\n");
    }

    if (y->grad) {
        printf("y->grad = %f (Expected: 2.0)\n", y->grad->data[0]);
    } else {
        printf("y->grad is NULL!\n");
    }

    // Cleanup
    // Freeing root of the graph (z) frees the intermediate nodes (MulBackward, AddBackward)
    // but we must manually free the leaf tensors and intermediate tensors we hold handles to.

    // z depends on sum and x.
    // sum depends on x and y.

    // When we free z, z->grad_fn (MulBackward) is freed.
    // When we free sum, sum->grad_fn (AddBackward) is freed.

    tensor_free(z);
    tensor_free(sum);
    tensor_free(y);
    tensor_free(x);

    printf("Experiment finished.\n");
    return EXIT_SUCCESS;
}
