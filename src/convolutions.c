#include "convolutions.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

//
// conv2d layer
//

typedef struct {
    Layer base;
    Tensor *weight;
    Tensor *bias;
    uint64_t in_channels;
    uint64_t out_channels;
    uint64_t kernel_size;
    uint64_t stride;
    uint64_t padding;
} Conv2dLayer;

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

static Tensor *conv2d_forward_impl(const Tensor *input, const Tensor *weight, const Tensor *bias, uint64_t stride, uint64_t padding, uint64_t kernel_size) {
    assert(input != NULL);
    assert(weight != NULL);

    uint64_t batch_size = input->shape[0];
    uint64_t in_channels = input->shape[1];
    uint64_t in_height = input->shape[2];
    uint64_t in_width = input->shape[3];

    uint64_t out_channels = weight->shape[0];
    uint64_t weight_in_channels = weight->shape[1];
    uint64_t weight_h = weight->shape[2];
    uint64_t weight_w = weight->shape[3];

    assert(in_channels == weight_in_channels && "input channels must match weight input channels");
    assert(weight_h == kernel_size && "weight height must match kernel size");
    assert(weight_w == kernel_size && "weight width must match kernel size");

    uint64_t out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    uint64_t out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

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

                    for (uint64_t k_h = 0; k_h < kernel_size; ++k_h) {
                        for (uint64_t k_w = 0; k_w < kernel_size; ++k_w) {
                            for (uint64_t in_ch = 0; in_ch < in_channels; ++in_ch) {
                                // input value
                                uint64_t in_idx = b * padded_input->strides[0] + in_ch * padded_input->strides[1] + (in_h_start + k_h) * padded_input->strides[2] + (in_w_start + k_w) * padded_input->strides[3];
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

    return output;
}

static Tensor *conv2d_forward(Layer *layer, const Tensor *input, bool training) {
    (void)training;
    const Conv2dLayer *l = (Conv2dLayer *)layer;
    assert(input->ndim == 4 && "input must be 4D tensor");
    return conv2d_forward_impl(input, l->weight, l->bias, l->stride, l->padding, l->kernel_size);
}

static void conv2d_free(Layer *layer) {
    Conv2dLayer *l = (Conv2dLayer *)layer;
    tensor_free(l->weight);
    if (l->bias) {
        tensor_free(l->bias);
    }
    free(l);
}

static void conv2d_parameters(Layer *layer, Tensor ***out_params, size_t *out_count) {
    Conv2dLayer *l = (Conv2dLayer *)layer;
    size_t count = (l->bias != NULL) ? 2 : 1;
    *out_params = malloc(sizeof(Tensor *) * count);
    assert(*out_params != NULL);
    (*out_params)[0] = l->weight;
    if (l->bias) {
        (*out_params)[1] = l->bias;
    }
    *out_count = count;
}

Layer *layer_conv2d_create(uint64_t in_channels, uint64_t out_channels, uint64_t kernel_size, uint64_t stride, uint64_t padding, bool bias) {
    Conv2dLayer *l = calloc(1, sizeof(Conv2dLayer));
    assert(l != NULL && "failed to allocate layer");
    l->base.forward = conv2d_forward;
    l->base.free = conv2d_free;
    l->base.parameters = conv2d_parameters;
    l->base.name = "Conv2d";

    l->in_channels = in_channels;
    l->out_channels = out_channels;
    l->kernel_size = kernel_size;
    l->stride = stride;
    l->padding = padding;

    // He init
    const uint64_t weight_shape[] = {out_channels, in_channels, kernel_size, kernel_size};
    uint64_t fan_in = in_channels * kernel_size * kernel_size;
    float32_t std = sqrtf(2.0f / (float32_t)fan_in);

    uint64_t w_size = out_channels * in_channels * kernel_size * kernel_size;
    float32_t *w_data = malloc(w_size * sizeof(float32_t));
    assert(w_data != NULL && "failed to allocate weight data");
    for (uint64_t i = 0; i < w_size; ++i) {
        // box muller for normal dist.
        float32_t u1 = (float32_t)rand() / (float32_t)RAND_MAX;
        float32_t u2 = (float32_t)rand() / (float32_t)RAND_MAX;
        if (u1 < 1e-6f) {
            u1 = 1e-6f;
        }
        float32_t z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float32_t)M_PI * u2);
        w_data[i] = z0 * std;
    }
    l->weight = tensor_create(w_data, weight_shape, 4, true);
    free(w_data);

    if (bias) {
        const uint64_t bias_shape[] = {out_channels};
        l->bias = tensor_zeros(bias_shape, 1, true);
    }

    return (Layer *)l;
}

//
// maxpool2d layer
//

typedef struct {
    Layer base;
    uint64_t kernel_size;
    uint64_t stride;
    uint64_t padding;
} MaxPool2dLayer;

static Tensor *maxpool2d_forward_impl(const Tensor *input, uint64_t kernel_size, uint64_t stride, uint64_t padding) {
    uint64_t batch_size = input->shape[0];
    uint64_t channels = input->shape[1];
    uint64_t in_height = input->shape[2];
    uint64_t in_width = input->shape[3];

    uint64_t out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    uint64_t out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

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
                    uint64_t in_h_start = out_h * stride;
                    uint64_t in_w_start = out_w * stride;

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
    return output;
}

static Tensor *maxpool2d_forward(Layer *layer, const Tensor *input, bool training) {
    (void)training;
    const MaxPool2dLayer *l = (MaxPool2dLayer *)layer;
    assert(input->ndim == 4 && "input must be 4D");
    return maxpool2d_forward_impl(input, l->kernel_size, l->stride, l->padding);
}

static void maxpool2d_free(Layer *layer) {
    if (layer) {
        free(layer);
    }
}

static void maxpool2d_parameters(Layer *layer, Tensor ***out_params, size_t *out_count) {
    (void)layer;
    *out_params = NULL;
    *out_count = 0;
}

Layer *layer_maxpool2d_create(uint64_t kernel_size, uint64_t stride, uint64_t padding) {
    MaxPool2dLayer *l = calloc(1, sizeof(MaxPool2dLayer));
    assert(l != NULL && "failed to allocate layer");
    l->base.forward = maxpool2d_forward;
    l->base.free = maxpool2d_free;
    l->base.parameters = maxpool2d_parameters;
    l->base.name = "MaxPool2d";

    l->kernel_size = kernel_size;
    l->stride = (stride == 0) ? kernel_size : stride; // default stride is kernel_size if 0
    l->padding = padding;

    return (Layer *)l;
}

//
// avgpool2d layer
//

typedef struct {
    Layer base;
    uint64_t kernel_size;
    uint64_t stride;
    uint64_t padding;
} AvgPool2dLayer;

static Tensor *avgpool2d_forward(Layer *layer, const Tensor *input, bool training) {
    (void)training;
    const AvgPool2dLayer *l = (AvgPool2dLayer *)layer;
    if (input->ndim != 4) {
        return NULL;
    }

    uint64_t kernel_size = l->kernel_size;
    uint64_t stride = l->stride;
    uint64_t padding = l->padding;

    uint64_t batch_size = input->shape[0];
    uint64_t channels = input->shape[1];
    uint64_t in_height = input->shape[2];
    uint64_t in_width = input->shape[3];

    uint64_t out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    uint64_t out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

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
                    uint64_t in_h_start = out_h * stride;
                    uint64_t in_w_start = out_w * stride;

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
    return output;
}

static void avgpool2d_free(Layer *layer) {
    if (layer) {
        free(layer);
    }
}

static void avgpool2d_parameters(Layer *layer, Tensor ***out_params, size_t *out_count) {
    (void)layer;
    *out_params = NULL;
    *out_count = 0;
}

Layer *layer_avgpool2d_create(uint64_t kernel_size, uint64_t stride, uint64_t padding) {
    AvgPool2dLayer *l = calloc(1, sizeof(AvgPool2dLayer));
    assert(l != NULL);
    l->base.forward = avgpool2d_forward;
    l->base.free = avgpool2d_free;
    l->base.parameters = avgpool2d_parameters;
    l->base.name = "AvgPool2d";

    l->kernel_size = kernel_size;
    l->stride = (stride == 0) ? kernel_size : stride;
    l->padding = padding;

    return (Layer *)l;
}

//
// batchnorm2d layer
//

typedef struct {
    Layer base;
    Tensor *gamma;
    Tensor *beta;
    Tensor *running_mean;
    Tensor *running_var;
    uint64_t num_features;
    float32_t eps;
    float32_t momentum;
} BatchNorm2dLayer;

static Tensor *batchnorm2d_forward_impl(Layer *layer, const Tensor *input, bool training) {
    BatchNorm2dLayer *l = (BatchNorm2dLayer *)layer;

    assert(input->ndim == 4 && "input must be 4D tensor");

    uint64_t batch_size = input->shape[0];
    uint64_t channels = input->shape[1];
    uint64_t height = input->shape[2];
    uint64_t width = input->shape[3];

    assert(channels == l->num_features && "input channels must match batchnorm num_features");

    bool requires_grad = input->requires_grad || l->gamma->requires_grad || l->beta->requires_grad;
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
            l->running_mean->data[c] = (1.0f - l->momentum) * l->running_mean->data[c] + l->momentum * mean;
            l->running_var->data[c] = (1.0f - l->momentum) * l->running_var->data[c] + l->momentum * var;
        }
    } else {
        // use running stats
        for (uint64_t c = 0; c < channels; ++c) {
            batch_mean[c] = l->running_mean->data[c];
            batch_var[c] = l->running_var->data[c];
        }
    }

    // normalize and scale/shift
    for (uint64_t c = 0; c < channels; ++c) {
        float32_t mean = batch_mean[c];
        float32_t var = batch_var[c];
        float32_t inv_std = 1.0f / sqrtf(var + l->eps);
        float32_t gamma = l->gamma->data[c];
        float32_t beta = l->beta->data[c];

        for (uint64_t b = 0; b < batch_size; ++b) {
            for (uint64_t h = 0; h < height; ++h) {
                for (uint64_t w = 0; w < width; ++w) {
                    uint64_t idx = b * input->strides[0] + c * input->strides[1] + h * input->strides[2] + w * input->strides[3];
                    float32_t val = input->data[idx];

                    uint64_t out_idx = b * output->strides[0] + c * output->strides[1] + h * output->strides[2] + w * output->strides[3];

                    // y = gamma * (x - mean) / sqrt(var + eps) + beta
                    output->data[out_idx] = gamma * (val - mean) * inv_std + beta;
                }
            }
        }
    }

    free(batch_mean);
    free(batch_var);

    return output;
}

static void batchnorm2d_free(Layer *layer) {
    BatchNorm2dLayer *l = (BatchNorm2dLayer *)layer;
    tensor_free(l->gamma);
    tensor_free(l->beta);
    tensor_free(l->running_mean);
    tensor_free(l->running_var);
    free(l);
}

static void batchnorm2d_parameters(Layer *layer, Tensor ***out_params, size_t *out_count) {
    BatchNorm2dLayer *l = (BatchNorm2dLayer *)layer;
    size_t count = 2; // gamma and beta
    *out_params = malloc(sizeof(Tensor *) * count);
    assert(*out_params != NULL);
    (*out_params)[0] = l->gamma;
    (*out_params)[1] = l->beta;
    *out_count = count;
}

Layer *layer_batchnorm2d_create(uint64_t num_features, float32_t eps, float32_t momentum) {
    BatchNorm2dLayer *l = calloc(1, sizeof(BatchNorm2dLayer));
    assert(l != NULL);
    l->base.forward = batchnorm2d_forward_impl;
    l->base.free = batchnorm2d_free;
    l->base.parameters = batchnorm2d_parameters;
    l->base.name = "BatchNorm2d";

    l->num_features = num_features;
    l->eps = eps;
    l->momentum = momentum;

    const uint64_t shape[] = {num_features};
    // gamma = 1
    l->gamma = tensor_create(NULL, shape, 1, true); // create creates uninitialized
    // fill gamma with 1
    for (size_t i = 0; i < l->gamma->size; ++i) {
        l->gamma->data[i] = 1.0f;
    }

    // beta = 0
    l->beta = tensor_zeros(shape, 1, true);

    // running_mean = 0 (no grad)
    l->running_mean = tensor_zeros(shape, 1, false);

    // running_var = 1 (no grad)
    l->running_var = tensor_create(NULL, shape, 1, false);
    for (size_t i = 0; i < l->running_var->size; ++i) {
        l->running_var->data[i] = 1.0f;
    }

    return (Layer *)l;
}
