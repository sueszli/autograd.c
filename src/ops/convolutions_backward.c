#include "convolutions.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// applies padding to a 4D tensor (batch_size, channels, height, width).
// returns a new tensor with shape (batch_size, channels, height+2p, width+2p).
// value is 0.0f (or -inf for maxpool if needed).
static Tensor *pad_tensor(const Tensor *input, uint64_t padding, float32_t value) {
    if (padding == 0) {
        return tensor_create(input->data, input->shape, input->ndim, false); // copy
    }

    uint64_t batch_size = input->shape[0];
    uint64_t channels = input->shape[1];
    uint64_t height = input->shape[2];
    uint64_t width = input->shape[3];

    uint64_t padded_height = height + 2 * padding;
    uint64_t padded_width = width + 2 * padding;

    const uint64_t out_shape[] = {batch_size, channels, padded_height, padded_width};
    Tensor *padded = tensor_create(NULL, out_shape, 4, false); // manual fill
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

void conv2d_backward(const Tensor *input, const Tensor *weight, const Tensor *bias, uint64_t stride, uint64_t padding, uint64_t kernel_size, const Tensor *grad_output, Tensor **out_grad_in, Tensor **out_grad_w, Tensor **out_grad_b) {
    assert(input != NULL);
    assert(weight != NULL);
    assert(grad_output != NULL);
    assert(out_grad_in != NULL);
    assert(out_grad_w != NULL);
    assert(out_grad_b != NULL);

    uint64_t batch_size = grad_output->shape[0];
    uint64_t out_channels = grad_output->shape[1];
    uint64_t out_height = grad_output->shape[2];
    uint64_t out_width = grad_output->shape[3];

    uint64_t in_channels = input->shape[1];

    // pad input
    Tensor *padded_input = pad_tensor(input, padding, 0.0f);
    assert(padded_input != NULL && "failed to allocate padded_input");

    // init gradients
    Tensor *grad_input_padded = tensor_zeros(padded_input->shape, 4, false);
    assert(grad_input_padded != NULL && "failed to allocate grad_input_padded");

    *out_grad_w = tensor_zeros(weight->shape, 4, false);
    assert(*out_grad_w != NULL && "failed to allocate out_grad_w");

    if (bias) {
        *out_grad_b = tensor_zeros(bias->shape, 1, false);
        assert(*out_grad_b != NULL && "failed to allocate out_grad_b");
    } else {
        *out_grad_b = NULL;
    }

    for (uint64_t b = 0; b < batch_size; ++b) {
        for (uint64_t out_ch = 0; out_ch < out_channels; ++out_ch) {
            for (uint64_t out_h = 0; out_h < out_height; ++out_h) {
                for (uint64_t out_w = 0; out_w < out_width; ++out_w) {
                    uint64_t in_h_start = out_h * stride;
                    uint64_t in_w_start = out_w * stride;

                    uint64_t grad_idx = b * grad_output->strides[0] + out_ch * grad_output->strides[1] + out_h * grad_output->strides[2] + out_w * grad_output->strides[3];
                    float32_t grad_val = grad_output->data[grad_idx];

                    for (uint64_t k_h = 0; k_h < kernel_size; ++k_h) {
                        for (uint64_t k_w = 0; k_w < kernel_size; ++k_w) {
                            for (uint64_t in_ch = 0; in_ch < in_channels; ++in_ch) {
                                // input pos details
                                uint64_t in_h = in_h_start + k_h;
                                uint64_t in_w = in_w_start + k_w;

                                // grad wrt. weight
                                uint64_t padded_idx = b * padded_input->strides[0] + in_ch * padded_input->strides[1] + in_h * padded_input->strides[2] + in_w * padded_input->strides[3];
                                float32_t val = padded_input->data[padded_idx];

                                uint64_t w_idx = out_ch * (*out_grad_w)->strides[0] + in_ch * (*out_grad_w)->strides[1] + k_h * (*out_grad_w)->strides[2] + k_w * (*out_grad_w)->strides[3];
                                (*out_grad_w)->data[w_idx] += val * grad_val;

                                // grad wrt. input
                                uint64_t w_val_idx = out_ch * weight->strides[0] + in_ch * weight->strides[1] + k_h * weight->strides[2] + k_w * weight->strides[3];
                                float32_t w_val = weight->data[w_val_idx];

                                uint64_t g_in_idx = b * grad_input_padded->strides[0] + in_ch * grad_input_padded->strides[1] + in_h * grad_input_padded->strides[2] + in_w * grad_input_padded->strides[3];
                                grad_input_padded->data[g_in_idx] += w_val * grad_val;
                            }
                        }
                    }
                }
            }
        }
    }

    // bias gradient
    if (*out_grad_b) {
        for (uint64_t out_ch = 0; out_ch < out_channels; ++out_ch) {
            float32_t sum = 0.0f;
            // sum over batch, height, width
            for (uint64_t b = 0; b < batch_size; ++b) {
                for (uint64_t h = 0; h < out_height; ++h) {
                    for (uint64_t w = 0; w < out_width; ++w) {
                        uint64_t idx = b * grad_output->strides[0] + out_ch * grad_output->strides[1] + h * grad_output->strides[2] + w * grad_output->strides[3];
                        sum += grad_output->data[idx];
                    }
                }
            }
            (*out_grad_b)->data[out_ch] = sum;
        }
    }

    // remove padding from input gradient
    if (padding == 0) {
        *out_grad_in = grad_input_padded;
        tensor_free(padded_input);
        return;
    }

    uint64_t inner_height = input->shape[2];
    uint64_t inner_width = input->shape[3];
    *out_grad_in = tensor_zeros(input->shape, 4, false);
    assert(*out_grad_in != NULL && "failed to allocate out_grad_in");

    for (uint64_t b = 0; b < batch_size; ++b) {
        for (uint64_t c = 0; c < in_channels; ++c) {
            for (uint64_t h = 0; h < inner_height; ++h) {
                for (uint64_t w = 0; w < inner_width; ++w) {
                    uint64_t src_idx = b * grad_input_padded->strides[0] + c * grad_input_padded->strides[1] + (h + padding) * grad_input_padded->strides[2] + (w + padding) * grad_input_padded->strides[3];
                    uint64_t dst_idx = b * (*out_grad_in)->strides[0] + c * (*out_grad_in)->strides[1] + h * (*out_grad_in)->strides[2] + w * (*out_grad_in)->strides[3];
                    (*out_grad_in)->data[dst_idx] = grad_input_padded->data[src_idx];
                }
            }
        }
    }
    tensor_free(grad_input_padded);
    tensor_free(padded_input);
}

Tensor *maxpool2d_backward(const Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding, const Tensor *grad_output) {
    uint64_t batch_size = input->shape[0];
    uint64_t channels = input->shape[1];
    uint64_t out_height = output_shape[2];
    uint64_t out_width = output_shape[3];

    Tensor *padded_input = pad_tensor(input, padding, -INFINITY);
    assert(padded_input != NULL && "failed to allocate padded_input");

    Tensor *grad_input_padded = tensor_zeros(padded_input->shape, 4, false);
    assert(grad_input_padded != NULL && "failed to allocate grad_input_padded");

    for (uint64_t b = 0; b < batch_size; ++b) {
        for (uint64_t c = 0; c < channels; ++c) {
            for (uint64_t out_h = 0; out_h < out_height; ++out_h) {
                for (uint64_t out_w = 0; out_w < out_width; ++out_w) {
                    uint64_t in_h_start = out_h * stride;
                    uint64_t in_w_start = out_w * stride;

                    float32_t max_val = -INFINITY;
                    uint64_t max_h = 0;
                    uint64_t max_w = 0;

                    // recompute max position
                    for (uint64_t k_h = 0; k_h < kernel_size; ++k_h) {
                        for (uint64_t k_w = 0; k_w < kernel_size; ++k_w) {
                            uint64_t in_h = in_h_start + k_h;
                            uint64_t in_w = in_w_start + k_w;

                            uint64_t in_idx = b * padded_input->strides[0] + c * padded_input->strides[1] + in_h * padded_input->strides[2] + in_w * padded_input->strides[3];
                            float32_t val = padded_input->data[in_idx];

                            if (val > max_val) {
                                max_val = val;
                                max_h = in_h;
                                max_w = in_w;
                            }
                        }
                    }

                    // route gradient
                    uint64_t grad_out_idx = b * grad_output->strides[0] + c * grad_output->strides[1] + out_h * grad_output->strides[2] + out_w * grad_output->strides[3];
                    float32_t g = grad_output->data[grad_out_idx];

                    uint64_t grad_in_idx = b * grad_input_padded->strides[0] + c * grad_input_padded->strides[1] + max_h * grad_input_padded->strides[2] + max_w * grad_input_padded->strides[3];
                    grad_input_padded->data[grad_in_idx] += g;
                }
            }
        }
    }

    if (padding == 0) {
        tensor_free(padded_input);
        return grad_input_padded;
    }

    uint64_t inner_height = input->shape[2];
    uint64_t inner_width = input->shape[3];
    Tensor *grad_input = tensor_zeros(input->shape, 4, false);
    assert(grad_input != NULL && "failed to allocate grad_input");

    for (uint64_t b = 0; b < batch_size; ++b) {
        for (uint64_t c = 0; c < channels; ++c) {
            for (uint64_t h = 0; h < inner_height; ++h) {
                for (uint64_t w = 0; w < inner_width; ++w) {
                    uint64_t src_idx = b * grad_input_padded->strides[0] + c * grad_input_padded->strides[1] + (h + padding) * grad_input_padded->strides[2] + (w + padding) * grad_input_padded->strides[3];
                    uint64_t dst_idx = b * grad_input->strides[0] + c * grad_input->strides[1] + h * grad_input->strides[2] + w * grad_input->strides[3];
                    grad_input->data[dst_idx] = grad_input_padded->data[src_idx];
                }
            }
        }
    }
    tensor_free(grad_input_padded);
    tensor_free(padded_input);
    return grad_input;
}

Tensor *avgpool2d_backward(const Tensor *input, const uint64_t *output_shape, uint64_t kernel_size, uint64_t stride, uint64_t padding, const Tensor *grad_output) {
    uint64_t batch_size = input->shape[0];
    uint64_t channels = input->shape[1];
    uint64_t out_height = output_shape[2];
    uint64_t out_width = output_shape[3];

    Tensor *padded_input = pad_tensor(input, padding, 0.0f);
    assert(padded_input != NULL && "failed to allocate padded_input");

    Tensor *grad_input_padded = tensor_zeros(padded_input->shape, 4, false);
    assert(grad_input_padded != NULL && "failed to allocate grad_input_padded");

    float32_t grad_scale = 1.0f / (float32_t)(kernel_size * kernel_size);

    for (uint64_t b = 0; b < batch_size; ++b) {
        for (uint64_t c = 0; c < channels; ++c) {
            for (uint64_t out_h = 0; out_h < out_height; ++out_h) {
                for (uint64_t out_w = 0; out_w < out_width; ++out_w) {
                    uint64_t in_h_start = out_h * stride;
                    uint64_t in_w_start = out_w * stride;

                    // get gradient from output
                    uint64_t grad_out_idx = b * grad_output->strides[0] + c * grad_output->strides[1] + out_h * grad_output->strides[2] + out_w * grad_output->strides[3];
                    float32_t g = grad_output->data[grad_out_idx];

                    // distribute gradient uniformly across kernel window
                    float32_t distributed_grad = g * grad_scale;

                    for (uint64_t k_h = 0; k_h < kernel_size; ++k_h) {
                        for (uint64_t k_w = 0; k_w < kernel_size; ++k_w) {
                            uint64_t in_h = in_h_start + k_h;
                            uint64_t in_w = in_w_start + k_w;

                            uint64_t grad_in_idx = b * grad_input_padded->strides[0] + c * grad_input_padded->strides[1] + in_h * grad_input_padded->strides[2] + in_w * grad_input_padded->strides[3];
                            grad_input_padded->data[grad_in_idx] += distributed_grad;
                        }
                    }
                }
            }
        }
    }

    if (padding == 0) {
        tensor_free(padded_input);
        return grad_input_padded;
    }

    uint64_t inner_height = input->shape[2];
    uint64_t inner_width = input->shape[3];
    Tensor *grad_input = tensor_zeros(input->shape, 4, false);
    assert(grad_input != NULL && "failed to allocate grad_input");

    for (uint64_t b = 0; b < batch_size; ++b) {
        for (uint64_t c = 0; c < channels; ++c) {
            for (uint64_t h = 0; h < inner_height; ++h) {
                for (uint64_t w = 0; w < inner_width; ++w) {
                    uint64_t src_idx = b * grad_input_padded->strides[0] + c * grad_input_padded->strides[1] + (h + padding) * grad_input_padded->strides[2] + (w + padding) * grad_input_padded->strides[3];
                    uint64_t dst_idx = b * grad_input->strides[0] + c * grad_input->strides[1] + h * grad_input->strides[2] + w * grad_input->strides[3];
                    grad_input->data[dst_idx] = grad_input_padded->data[src_idx];
                }
            }
        }
    }
    tensor_free(grad_input_padded);
    tensor_free(padded_input);
    return grad_input;
}

void batchnorm2d_backward(const Tensor *input, const Tensor *gamma, const Tensor *batch_mean, const Tensor *batch_var, float32_t eps, const Tensor *grad_output, Tensor **out_grad_in, Tensor **out_grad_gamma, Tensor **out_grad_beta) {
    assert(input != NULL);
    assert(gamma != NULL);
    assert(batch_mean != NULL);
    assert(batch_var != NULL);
    assert(grad_output != NULL);
    assert(out_grad_in != NULL);
    assert(out_grad_gamma != NULL);
    assert(out_grad_beta != NULL);

    uint64_t batch_size = input->shape[0];
    uint64_t channels = input->shape[1];
    uint64_t height = input->shape[2];
    uint64_t width = input->shape[3];

    uint64_t n_pixels = batch_size * height * width;

    // allocate gradients
    *out_grad_in = tensor_zeros(input->shape, 4, false);
    assert(*out_grad_in != NULL && "failed to allocate out_grad_in");

    const uint64_t param_shape[] = {channels};
    *out_grad_gamma = tensor_zeros(param_shape, 1, false);
    assert(*out_grad_gamma != NULL && "failed to allocate out_grad_gamma");

    *out_grad_beta = tensor_zeros(param_shape, 1, false);
    assert(*out_grad_beta != NULL && "failed to allocate out_grad_beta");

    // per-channel computation
    for (uint64_t c = 0; c < channels; ++c) {
        float32_t mean = batch_mean->data[c];
        float32_t var = batch_var->data[c];
        float32_t inv_std = 1.0f / sqrtf(var + eps);
        float32_t gamma_val = gamma->data[c];

        // compute grad_beta and grad_gamma
        float32_t grad_beta_sum = 0.0f;
        float32_t grad_gamma_sum = 0.0f;

        for (uint64_t b = 0; b < batch_size; ++b) {
            for (uint64_t h = 0; h < height; ++h) {
                for (uint64_t w = 0; w < width; ++w) {
                    uint64_t idx = b * input->strides[0] + c * input->strides[1] + h * input->strides[2] + w * input->strides[3];
                    float32_t x = input->data[idx];
                    float32_t x_hat = (x - mean) * inv_std;

                    uint64_t grad_idx = b * grad_output->strides[0] + c * grad_output->strides[1] + h * grad_output->strides[2] + w * grad_output->strides[3];
                    float32_t grad_out = grad_output->data[grad_idx];

                    grad_beta_sum += grad_out;
                    grad_gamma_sum += grad_out * x_hat;
                }
            }
        }

        (*out_grad_beta)->data[c] = grad_beta_sum;
        (*out_grad_gamma)->data[c] = grad_gamma_sum;

        // compute grad_input using standard batchnorm backward formula
        // grad_input = gamma / sqrt(var + eps) * [grad_output - mean(grad_output) - x_hat * mean(grad_output * x_hat)]
        float32_t grad_out_mean = grad_beta_sum / (float32_t)n_pixels;
        float32_t grad_out_x_hat_mean = grad_gamma_sum / (float32_t)n_pixels;

        for (uint64_t b = 0; b < batch_size; ++b) {
            for (uint64_t h = 0; h < height; ++h) {
                for (uint64_t w = 0; w < width; ++w) {
                    uint64_t idx = b * input->strides[0] + c * input->strides[1] + h * input->strides[2] + w * input->strides[3];
                    float32_t x = input->data[idx];
                    float32_t x_hat = (x - mean) * inv_std;

                    uint64_t grad_idx = b * grad_output->strides[0] + c * grad_output->strides[1] + h * grad_output->strides[2] + w * grad_output->strides[3];
                    float32_t grad_out = grad_output->data[grad_idx];

                    float32_t grad_x_hat = grad_out - grad_out_mean - x_hat * grad_out_x_hat_mean;
                    float32_t grad_x = gamma_val * inv_std * grad_x_hat;

                    uint64_t out_idx = b * (*out_grad_in)->strides[0] + c * (*out_grad_in)->strides[1] + h * (*out_grad_in)->strides[2] + w * (*out_grad_in)->strides[3];
                    (*out_grad_in)->data[out_idx] = grad_x;
                }
            }
        }
    }
}
