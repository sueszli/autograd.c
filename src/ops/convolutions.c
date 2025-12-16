#include "convolutions.h"
#include "autograd.h"
#include "convolutions_backward.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// applies padding to a 4D tensor (batch_size, channels, height, width).
// returns a new tensor with shape (batch_size, channels, height+2p, width+2p).
// value is 0.0f (or -inf for maxpool if needed).
static Tensor *pad_tensor(const Tensor *input, uint64_t padding, float32_t value) {
    if (padding == 0) {
        return tensor_create(input->data, input->shape, input->ndim, input->requires_grad); // copy
    }

    uint64_t batch_size = input->shape[0];
    uint64_t channels = input->shape[1];
    uint64_t height = input->shape[2];
    uint64_t width = input->shape[3];

    uint64_t padded_height = height + 2 * padding;
    uint64_t padded_width = width + 2 * padding;

    const uint64_t out_shape[] = {batch_size, channels, padded_height, padded_width};
    Tensor *padded = tensor_create(NULL, out_shape, 4, input->requires_grad);
    assert(padded != NULL && "failed to allocate padded tensor");

    // fill with value
    for (uint64_t i = 0; i < padded->size; ++i) {
        padded->data[i] = value;
    }

    // copy input into center
    for (uint64_t b = 0; b < batch_size; ++b) {
        for (uint64_t c = 0; c < channels; ++c) {
            for (uint64_t h = 0; h < height; ++h) {
                for (uint64_t w = 0; w < width; ++w) {
                    uint64_t in_idx = b * input->strides[0] + c * input->strides[1] + h * input->strides[2] + w * input->strides[3];

                    uint64_t out_idx = b * padded->strides[0] + c * padded->strides[1] + (h + padding) * padded->strides[2] + (w + padding) * padded->strides[3];

                    padded->data[out_idx] = input->data[in_idx];
                }
            }
        }
    }

    return padded;
}

Tensor *tensor_conv2d(const Tensor *input, const Tensor *weight, const Tensor *bias, uint64_t stride, uint64_t padding, uint64_t dilation) {
    assert(input != NULL);
    assert(weight != NULL);

    uint64_t batch_size = input->shape[0];
    uint64_t in_channels = input->shape[1];
    uint64_t in_height = input->shape[2];
    uint64_t in_width = input->shape[3];

    uint64_t out_channels = weight->shape[0];
    uint64_t weight_in_channels = weight->shape[1];
    (void)weight_in_channels;
    uint64_t kernel_h = weight->shape[2];
    uint64_t kernel_w = weight->shape[3];

    assert(in_channels == weight_in_channels && "input channels must match weight input channels");

    // support dilation (effective kernel size)
    uint64_t eff_kernel_h = dilation * (kernel_h - 1) + 1;
    uint64_t eff_kernel_w = dilation * (kernel_w - 1) + 1;

    uint64_t out_height = (in_height + 2 * padding - eff_kernel_h) / stride + 1;
    uint64_t out_width = (in_width + 2 * padding - eff_kernel_w) / stride + 1;

    const uint64_t out_shape[] = {batch_size, out_channels, out_height, out_width};
    bool requires_grad = input->requires_grad || weight->requires_grad || (bias && bias->requires_grad);
    Tensor *output = tensor_zeros(out_shape, 4, requires_grad);
    assert(output != NULL && "failed to allocate output tensor");

    // apply padding
    Tensor *padded_input = pad_tensor(input, padding, 0.0f);
    assert(padded_input != NULL && "failed to allocate padded input");

    for (uint64_t b = 0; b < batch_size; ++b) {
        for (uint64_t out_ch = 0; out_ch < out_channels; ++out_ch) {
            float32_t bias_val = (bias != NULL) ? bias->data[out_ch] : 0.0f;

            for (uint64_t out_h = 0; out_h < out_height; ++out_h) {
                for (uint64_t out_w = 0; out_w < out_width; ++out_w) {
                    uint64_t in_h_start = out_h * stride;
                    uint64_t in_w_start = out_w * stride;

                    float32_t conv_sum = 0.0f;

                    for (uint64_t k_h = 0; k_h < kernel_h; ++k_h) {
                        for (uint64_t k_w = 0; k_w < kernel_w; ++k_w) {
                            for (uint64_t in_ch = 0; in_ch < in_channels; ++in_ch) {
                                // input value with dilation
                                uint64_t h_offset = k_h * dilation;
                                uint64_t w_offset = k_w * dilation;

                                uint64_t in_idx = b * padded_input->strides[0] + in_ch * padded_input->strides[1] + (in_h_start + h_offset) * padded_input->strides[2] + (in_w_start + w_offset) * padded_input->strides[3];
                                float32_t val = padded_input->data[in_idx];

                                // weight value
                                uint64_t w_idx = out_ch * weight->strides[0] + in_ch * weight->strides[1] + k_h * weight->strides[2] + k_w * weight->strides[3];
                                float32_t w_val = weight->data[w_idx];

                                conv_sum += val * w_val;
                            }
                        }
                    }

                    uint64_t out_idx = b * output->strides[0] + out_ch * output->strides[1] + out_h * output->strides[2] + out_w * output->strides[3];
                    output->data[out_idx] = conv_sum + bias_val;
                }
            }
        }
    }

    tensor_free(padded_input);

    if (output->requires_grad) {
        Function *fn = arena_alloc_function();
        fn->apply = conv2d_backward_fn;
        fn->output = output;

        uint32_t num_inputs = 0;
        fn->inputs[num_inputs++] = (Tensor *)input;
        fn->inputs[num_inputs++] = (Tensor *)weight;
        if (bias != NULL) {
            fn->inputs[num_inputs++] = (Tensor *)bias;
        }
        fn->num_inputs = num_inputs;
        fn->pending_count = 0;

        Conv2dContext *ctx = (Conv2dContext *)malloc(sizeof(Conv2dContext));
        assert(ctx != NULL && "malloc failed");
        ctx->stride = stride;
        ctx->padding = padding;
        ctx->dilation = dilation;
        ctx->kernel_h = kernel_h;
        ctx->kernel_w = kernel_w;
        fn->ctx = ctx;

        if (input->grad_fn != NULL) {
            input->grad_fn->pending_count++;
        }
        if (weight->grad_fn != NULL) {
            weight->grad_fn->pending_count++;
        }
        if (bias && bias->grad_fn != NULL) {
            bias->grad_fn->pending_count++;
        }

        output->grad_fn = fn;
    }

    return output;
}

Tensor *tensor_maxpool2d(const Tensor *input, uint64_t kernel_size, uint64_t stride, uint64_t padding) {
    uint64_t batch_size = input->shape[0];
    uint64_t channels = input->shape[1];
    uint64_t in_height = input->shape[2];
    uint64_t in_width = input->shape[3];

    uint64_t eff_stride = (stride == 0) ? kernel_size : stride;

    uint64_t out_height = (in_height + 2 * padding - kernel_size) / eff_stride + 1;
    uint64_t out_width = (in_width + 2 * padding - kernel_size) / eff_stride + 1;

    const uint64_t out_shape[] = {batch_size, channels, out_height, out_width};
    Tensor *output = tensor_zeros(out_shape, 4, input->requires_grad);
    assert(output != NULL && "failed to allocate output tensor");

    // apply padding
    // for MaxPool, padding value should be -inf
    Tensor *padded_input = pad_tensor(input, padding, -INFINITY);
    assert(padded_input != NULL && "failed to allocate padded_input");

    for (uint64_t b = 0; b < batch_size; ++b) {
        for (uint64_t c = 0; c < channels; ++c) {
            for (uint64_t out_h = 0; out_h < out_height; ++out_h) {
                for (uint64_t out_w = 0; out_w < out_width; ++out_w) {
                    uint64_t in_h_start = out_h * eff_stride;
                    uint64_t in_w_start = out_w * eff_stride;

                    float32_t max_val = -INFINITY;

                    for (uint64_t k_h = 0; k_h < kernel_size; ++k_h) {
                        for (uint64_t k_w = 0; k_w < kernel_size; ++k_w) {
                            uint64_t in_idx = b * padded_input->strides[0] + c * padded_input->strides[1] + (in_h_start + k_h) * padded_input->strides[2] + (in_w_start + k_w) * padded_input->strides[3];
                            float32_t val = padded_input->data[in_idx];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }

                    uint64_t out_idx = b * output->strides[0] + c * output->strides[1] + out_h * output->strides[2] + out_w * output->strides[3];
                    output->data[out_idx] = max_val;
                }
            }
        }
    }

    tensor_free(padded_input);

    if (output->requires_grad) {
        Function *fn = arena_alloc_function();
        fn->apply = maxpool2d_backward_fn;
        fn->output = output;
        fn->num_inputs = 1;
        fn->inputs[0] = (Tensor *)input;
        fn->pending_count = 0;

        MaxPool2dContext *ctx = (MaxPool2dContext *)malloc(sizeof(MaxPool2dContext));
        assert(ctx != NULL && "malloc failed");
        ctx->kernel_size = kernel_size;
        ctx->stride = eff_stride;
        ctx->padding = padding;
        ctx->output_shape[0] = output->shape[0];
        ctx->output_shape[1] = output->shape[1];
        ctx->output_shape[2] = output->shape[2];
        ctx->output_shape[3] = output->shape[3];
        fn->ctx = ctx;

        if (input->grad_fn != NULL) {
            input->grad_fn->pending_count++;
        }

        output->grad_fn = fn;
    }

    return output;
}

Tensor *tensor_avgpool2d(const Tensor *input, uint64_t kernel_size, uint64_t stride, uint64_t padding) {
    if (input->ndim != 4) {
        return NULL;
    }

    uint64_t batch_size = input->shape[0];
    uint64_t channels = input->shape[1];
    uint64_t in_height = input->shape[2];
    uint64_t in_width = input->shape[3];

    uint64_t eff_stride = (stride == 0) ? kernel_size : stride;

    uint64_t out_height = (in_height + 2 * padding - kernel_size) / eff_stride + 1;
    uint64_t out_width = (in_width + 2 * padding - kernel_size) / eff_stride + 1;

    const uint64_t out_shape[] = {batch_size, channels, out_height, out_width};
    Tensor *output = tensor_zeros(out_shape, 4, input->requires_grad);
    assert(output != NULL && "failed to allocate output tensor");

    // apply padding
    Tensor *padded_input = pad_tensor(input, padding, 0.0f);
    assert(padded_input != NULL && "failed to allocate padded_input");

    for (uint64_t b = 0; b < batch_size; ++b) {
        for (uint64_t c = 0; c < channels; ++c) {
            for (uint64_t out_h = 0; out_h < out_height; ++out_h) {
                for (uint64_t out_w = 0; out_w < out_width; ++out_w) {
                    uint64_t in_h_start = out_h * eff_stride;
                    uint64_t in_w_start = out_w * eff_stride;

                    float32_t window_sum = 0.0f;

                    for (uint64_t k_h = 0; k_h < kernel_size; ++k_h) {
                        for (uint64_t k_w = 0; k_w < kernel_size; ++k_w) {
                            uint64_t in_idx = b * padded_input->strides[0] + c * padded_input->strides[1] + (in_h_start + k_h) * padded_input->strides[2] + (in_w_start + k_w) * padded_input->strides[3];
                            window_sum += padded_input->data[in_idx];
                        }
                    }

                    float32_t avg_val = window_sum / (float32_t)(kernel_size * kernel_size);

                    uint64_t out_idx = b * output->strides[0] + c * output->strides[1] + out_h * output->strides[2] + out_w * output->strides[3];
                    output->data[out_idx] = avg_val;
                }
            }
        }
    }

    tensor_free(padded_input);

    if (output->requires_grad) {
        Function *fn = arena_alloc_function();
        fn->apply = avgpool2d_backward_fn;
        fn->output = output;
        fn->num_inputs = 1;
        fn->inputs[0] = (Tensor *)input;
        fn->pending_count = 0;

        AvgPool2dContext *ctx = (AvgPool2dContext *)malloc(sizeof(AvgPool2dContext));
        assert(ctx != NULL && "malloc failed");
        ctx->kernel_size = kernel_size;
        ctx->stride = eff_stride;
        ctx->padding = padding;
        ctx->output_shape[0] = output->shape[0];
        ctx->output_shape[1] = output->shape[1];
        ctx->output_shape[2] = output->shape[2];
        ctx->output_shape[3] = output->shape[3];
        fn->ctx = ctx;

        if (input->grad_fn != NULL) {
            input->grad_fn->pending_count++;
        }

        output->grad_fn = fn;
    }

    return output;
}

Tensor *tensor_batchnorm2d(const Tensor *input, const Tensor *gamma, const Tensor *beta, Tensor *running_mean, Tensor *running_var, bool training, float32_t momentum, float32_t eps) {
    assert(input->ndim == 4 && "input must be 4D tensor");

    uint64_t batch_size = input->shape[0];
    uint64_t channels = input->shape[1];
    uint64_t height = input->shape[2];
    uint64_t width = input->shape[3];

    assert(gamma->size == channels && "gamma size must match input channels");
    assert(beta->size == channels && "beta size must match input channels");
    assert(running_mean->size == channels && "running_mean size must match input channels");
    assert(running_var->size == channels && "running_var size must match input channels");

    bool requires_grad = input->requires_grad || gamma->requires_grad || beta->requires_grad;
    Tensor *output = tensor_zeros(input->shape, 4, requires_grad);
    assert(output != NULL && "failed to allocate output tensor");

    // per-channel mean and var
    float32_t *batch_mean = calloc(channels, sizeof(float32_t));
    assert(batch_mean != NULL && "failed to allocate batch_mean");
    float32_t *batch_var = calloc(channels, sizeof(float32_t));
    assert(batch_var != NULL && "failed to allocate batch_var");

    if (training) {
        // compute batch stats
        uint64_t n_pixels = batch_size * height * width;

        for (uint64_t c = 0; c < channels; ++c) {
            float32_t sum = 0.0f;
            float32_t sq_sum = 0.0f;

            for (uint64_t b = 0; b < batch_size; ++b) {
                for (uint64_t h = 0; h < height; ++h) {
                    for (uint64_t w = 0; w < width; ++w) {
                        uint64_t idx = b * input->strides[0] + c * input->strides[1] + h * input->strides[2] + w * input->strides[3];
                        float32_t val = input->data[idx];
                        sum += val;
                        sq_sum += val * val;
                    }
                }
            }

            float32_t mean = sum / (float32_t)n_pixels;
            // Var = E[x^2] - (E[x])^2
            float32_t mean_sq = sq_sum / (float32_t)n_pixels;
            float32_t var = mean_sq - mean * mean;

            batch_mean[c] = mean;
            batch_var[c] = var;

            // update running stats
            // running = (1 - momentum) * running + momentum * batch
            running_mean->data[c] = (1.0f - momentum) * running_mean->data[c] + momentum * mean;
            running_var->data[c] = (1.0f - momentum) * running_var->data[c] + momentum * var;
        }
    } else {
        // use running stats
        for (uint64_t c = 0; c < channels; ++c) {
            batch_mean[c] = running_mean->data[c];
            batch_var[c] = running_var->data[c];
        }
    }

    // normalize and scale/shift
    for (uint64_t c = 0; c < channels; ++c) {
        float32_t mean = batch_mean[c];
        float32_t var = batch_var[c];
        float32_t inv_std = 1.0f / sqrtf(var + eps);
        float32_t gamma_val = gamma->data[c];
        float32_t beta_val = beta->data[c];

        for (uint64_t b = 0; b < batch_size; ++b) {
            for (uint64_t h = 0; h < height; ++h) {
                for (uint64_t w = 0; w < width; ++w) {
                    uint64_t idx = b * input->strides[0] + c * input->strides[1] + h * input->strides[2] + w * input->strides[3];
                    float32_t val = input->data[idx];

                    uint64_t out_idx = b * output->strides[0] + c * output->strides[1] + h * output->strides[2] + w * output->strides[3];

                    // y = gamma * (x - mean) / sqrt(var + eps) + beta
                    output->data[out_idx] = gamma_val * (val - mean) * inv_std + beta_val;
                }
            }
        }
    }

    if (output->requires_grad) {
        Function *fn = arena_alloc_function();
        fn->apply = batchnorm2d_backward_fn;
        fn->output = output;

        fn->num_inputs = 3;
        fn->inputs[0] = (Tensor *)input;
        fn->inputs[1] = (Tensor *)gamma;
        fn->inputs[2] = (Tensor *)beta;
        fn->pending_count = 0;

        BatchNorm2dContext *ctx = (BatchNorm2dContext *)malloc(sizeof(BatchNorm2dContext));
        assert(ctx != NULL && "malloc failed");
        ctx->eps = eps;
        ctx->training = training;

        // store batch_mean and batch_var as tensors for backward
        const uint64_t param_shape[] = {channels};
        ctx->batch_mean = tensor_create(batch_mean, param_shape, 1, false);
        assert(ctx->batch_mean != NULL && "failed to allocate batch_mean tensor");
        ctx->batch_var = tensor_create(batch_var, param_shape, 1, false);
        assert(ctx->batch_var != NULL && "failed to allocate batch_var tensor");

        fn->ctx = ctx;

        if (input->grad_fn != NULL) {
            input->grad_fn->pending_count++;
        }
        if (gamma->grad_fn != NULL) {
            gamma->grad_fn->pending_count++;
        }
        if (beta->grad_fn != NULL) {
            beta->grad_fn->pending_count++;
        }

        output->grad_fn = fn;
    }

    free(batch_mean);
    free(batch_var);

    return output;
}
