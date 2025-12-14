#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("=== Tensor Library Demo ===\n\n");

    // 1. Create Tensor A (2x3)
    float32_t data_a[] = {1.0f, 2.0f, 3.0f, 
                          4.0f, 5.0f, 6.0f};
    uint64_t shape_a[] = {2, 3};
    Tensor *a = tensor_create(data_a, shape_a, 2, false);
    
    printf("Tensor A:\n");
    tensor_print(a);
    printf("\n");

    // 2. Create Tensor B (2x3)
    float32_t data_b[] = {6.0f, 5.0f, 4.0f, 
                          3.0f, 2.0f, 1.0f};
    uint64_t shape_b[] = {2, 3};
    Tensor *b = tensor_create(data_b, shape_b, 2, false);

    printf("Tensor B:\n");
    tensor_print(b);
    printf("\n");

    // 3. Element-wise Addition: C = A + B
    Tensor *c = tensor_add(a, b);
    printf("C = A + B:\n");
    tensor_print(c);
    printf("\n");

    // 4. Element-wise Multiplication: D = A * C
    Tensor *d = tensor_mul(a, c);
    printf("D = A * C:\n");
    tensor_print(d);
    printf("\n");

    // 5. Matrix Multiplication: E = A @ Transpose(B)
    // A is [2, 3], B is [2, 3]. To matmul, we transpose B to [3, 2].
    Tensor *b_t = tensor_transpose(b, 0, 1); // Swap dim 0 and 1
    printf("Transpose of B (B.T):\n");
    tensor_print(b_t);
    printf("\n");

    Tensor *e = tensor_matmul(a, b_t); // [2, 3] @ [3, 2] -> [2, 2]
    printf("E = A @ B.T:\n");
    tensor_print(e);
    printf("\n");

    // 6. Reduction: Sum of E
    Tensor *sum_e = tensor_sum(e, 0, false); // Sum along dim 0
    printf("Sum of E along dim 0:\n");
    tensor_print(sum_e);
    printf("\n");

    // Cleanup
    tensor_free(a);
    tensor_free(b);
    tensor_free(c);
    tensor_free(d);
    tensor_free(b_t);
    tensor_free(e);
    tensor_free(sum_e);

    return 0;
}
