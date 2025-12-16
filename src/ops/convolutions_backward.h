#pragma once

#include "autograd.h"
#include "layers.h"
#include "tensor.h"

void conv2d_backward(const Tensor *input, const Tensor *weight, const Tensor *bias, uint64_t stride, uint64_t padding, uint64_t kernel_size, const Tensor *grad_output, Tensor **out_grad_in, Tensor **out_grad_w, Tensor **out_grad_b);
typedef struct {
    uint64_t stride;
    uint64_t padding;
    uint64_t dilation;
    uint64_t kernel_h;
    uint64_t kernel_w;
} Conv2dContext;
void conv2d_backward_fn(Function *fn, const Tensor *grad_output);

Tensor *maxpool2d_backward(const Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding, const Tensor *grad_output);
typedef struct {
    uint64_t kernel_size;
    uint64_t stride;
    uint64_t padding;
    uint64_t output_shape[4];
} MaxPool2dContext;
void maxpool2d_backward_fn(Function *fn, const Tensor *grad_output);

Tensor *avgpool2d_backward(const Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding, const Tensor *grad_output);
typedef struct {
    uint64_t kernel_size;
    uint64_t stride;
    uint64_t padding;
    uint64_t output_shape[4];
} AvgPool2dContext;
void avgpool2d_backward_fn(Function *fn, const Tensor *grad_output);

void batchnorm2d_backward(const Tensor *input, const Tensor *gamma, const Tensor *batch_mean, const Tensor *batch_var, float32_t eps, const Tensor *grad_output, Tensor **out_grad_in, Tensor **out_grad_gamma, Tensor **out_grad_beta);
typedef struct {
    float32_t eps;
    bool training;
    Tensor *batch_mean;
    Tensor *batch_var;
} BatchNorm2dContext;
void batchnorm2d_backward_fn(Function *fn, const Tensor *grad_output);
