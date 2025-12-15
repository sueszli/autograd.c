#include "layers.h"
#include "tensor.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//
// base layer implementation
//

Tensor *layer_forward(Layer *layer, const Tensor *input, bool training) {
    assert(layer != NULL);
    assert(layer->forward != NULL);
    assert(input != NULL);

    return layer->forward(layer, input, training);
}

void layer_free(Layer *layer) {
    if (layer != NULL && layer->free != NULL) {
        layer->free(layer);
    }
}

void layer_parameters(Layer *layer, Tensor ***out_params, size_t *out_count) {
    assert(out_params != NULL);
    assert(out_count != NULL);

    if (layer != NULL && layer->parameters != NULL) {
        layer->parameters(layer, out_params, out_count);
        return;
    }
    *out_params = NULL;
    *out_count = 0;
}

//
// linear layer
//

typedef struct {
    Layer base;
    Tensor *weight;
    Tensor *bias;
    uint64_t in_features;
    uint64_t out_features;
} LinearLayer;

static Tensor *linear_forward(Layer *layer, const Tensor *input, bool training) {
    assert(layer != NULL);
    assert(input != NULL);
    (void)training;

    const LinearLayer *l = (const LinearLayer *)layer;
    assert(l->weight != NULL);

    // output = input @ weight + bias
    Tensor *output = tensor_matmul(input, l->weight);
    assert(output != NULL);

    if (l->bias != NULL) {
        Tensor *output_bias = tensor_add(output, l->bias);
        assert(output_bias != NULL);
        tensor_free(output);
        output = output_bias;
    }

    return output;
}

static void linear_free(Layer *layer) {
    if (layer == NULL) {
        return;
    }

    LinearLayer *l = (LinearLayer *)layer;
    if (l->weight != NULL) {
        tensor_free(l->weight);
    }
    if (l->bias != NULL) {
        tensor_free(l->bias);
    }
    free(l);
}

static void linear_parameters(Layer *layer, Tensor ***out_params, size_t *out_count) {
    assert(layer != NULL);
    assert(out_params != NULL);
    assert(out_count != NULL);

    LinearLayer *l = (LinearLayer *)layer;
    size_t const count = (l->bias != NULL) ? 2 : 1;

    Tensor **params = (Tensor **)malloc(count * sizeof(Tensor *));
    assert(params != NULL && "malloc failed");

    params[0] = l->weight;
    if (l->bias != NULL) {
        params[1] = l->bias;
    }
    *out_params = params;
    *out_count = count;
}

Layer *layer_linear_create(uint64_t in_features, uint64_t features_out, bool bias) {
    assert(in_features > 0);
    assert(features_out > 0);
    assert(in_features < (SIZE_MAX / features_out));

    LinearLayer *l = (LinearLayer *)calloc(1, sizeof(LinearLayer));
    assert(l != NULL && "calloc failed");

    l->base.forward = linear_forward;
    l->base.free = linear_free;
    l->base.parameters = linear_parameters;
    l->base.name = "Linear";
    l->in_features = in_features;
    l->out_features = features_out;

    float32_t const limit = 1.0f / sqrtf((float32_t)in_features);
    uint64_t const w_shape[] = {in_features, features_out};
    uint64_t const w_size = in_features * features_out;
    float32_t *w_data = (float32_t *)malloc(w_size * sizeof(float32_t));
    assert(w_data != NULL && "malloc failed");

    for (size_t i = 0; i < w_size; ++i) {
        float32_t const r = (float32_t)rand() / (float32_t)RAND_MAX;
        w_data[i] = (r * 2.0f * limit) - limit;
    }

    l->weight = tensor_create(w_data, w_shape, 2, true);
    assert(l->weight != NULL);
    free(w_data);

    if (bias) {
        uint64_t const b_shape[] = {features_out};
        l->bias = tensor_zeros(b_shape, 1, true);
        assert(l->bias != NULL);
    } else {
        l->bias = NULL;
    }

    return (Layer *)l;
}

//
// dropout layer
//

typedef struct {
    Layer base;
    float32_t p;
} DropoutLayer;

static Tensor *dropout_forward(const Layer *layer, const Tensor *input, bool training) {
    assert(layer != NULL);
    assert(input != NULL);

    const DropoutLayer *l = (const DropoutLayer *)layer;
    if (!training || l->p <= 0.0f) {
        Tensor *t = tensor_create(input->data, input->shape, input->ndim, input->requires_grad);
        assert(t != NULL);
        return t;
    }

    if (l->p >= 1.0f) {
        Tensor *t = tensor_zeros(input->shape, input->ndim, input->requires_grad);
        assert(t != NULL);
        return t;
    }

    float32_t const scale = 1.0f / (1.0f - l->p);
    float32_t *mask_data = (float32_t *)malloc(input->size * sizeof(float32_t));
    assert(mask_data != NULL && "malloc failed");

    // mask_data[i] = 0 with probability p, or 1/(1-p) with probability (1-p)
    for (size_t i = 0; i < input->size; ++i) {
        float32_t const r = (float32_t)rand() / (float32_t)RAND_MAX;
        mask_data[i] = (r < (1.0f - l->p)) ? scale : 0.0f;
    }

    Tensor *mask = tensor_create(mask_data, input->shape, input->ndim, false);
    assert(mask != NULL);
    free(mask_data);

    // output = input * mask
    Tensor *output = tensor_mul(input, mask);
    assert(output != NULL);
    tensor_free(mask);

    return output;
}

static void dropout_free(Layer *layer) {
    if (layer == NULL) {
        return;
    }
    free(layer);
}

static void dropout_parameters(Layer *layer, Tensor ***out_params, size_t *out_count) {
    (void)layer;
    assert(out_params != NULL);
    assert(out_count != NULL);
    *out_params = NULL;
    *out_count = 0;
}

Layer *layer_dropout_create(float32_t p) {
    assert(p >= 0.0f && p <= 1.0f && "dropout probability must be between 0 and 1");

    DropoutLayer *l = (DropoutLayer *)calloc(1, sizeof(DropoutLayer));
    assert(l != NULL && "calloc failed");

    l->base.forward = (Tensor * (*)(Layer *, const Tensor *, bool)) dropout_forward;
    l->base.free = dropout_free;
    l->base.parameters = dropout_parameters;
    l->base.name = "Dropout";
    l->p = p;
    return (Layer *)l;
}

//
// sequential layer
//

typedef struct {
    Layer base;
    Layer **layers;
    size_t count;
} SequentialLayer;

static Tensor *sequential_forward(Layer *layer, const Tensor *input, bool training) {
    assert(layer != NULL);
    assert(input != NULL);

    const SequentialLayer *l = (const SequentialLayer *)layer;

    if (l->count == 0) {
        Tensor *t = tensor_create(input->data, input->shape, input->ndim, input->requires_grad);
        assert(t != NULL);
        return t;
    }

    Tensor *current = NULL;

    // first layer
    current = layer_forward(l->layers[0], input, training);
    assert(current != NULL);

    for (size_t i = 1; i < l->count; ++i) {
        // composition: pass output of previous layer as input to next layer
        Tensor *next = layer_forward(l->layers[i], current, training);
        assert(next != NULL);
        tensor_free(current);
        current = next;
    }

    return current;
}

static void sequential_free(Layer *layer) {
    if (layer == NULL) {
        return;
    }

    SequentialLayer *l = (SequentialLayer *)layer;
    for (size_t i = 0; i < l->count; ++i) {
        layer_free(l->layers[i]);
    }
    free(l->layers);
    free(l);
}

static void sequential_parameters(Layer *layer, Tensor ***out_params, size_t *out_count) {
    assert(layer != NULL);
    assert(out_params != NULL);
    assert(out_count != NULL);

    SequentialLayer *l = (SequentialLayer *)layer;

    size_t total_params = 0;
    for (size_t i = 0; i < l->count; ++i) {
        Tensor **sub_params;
        size_t sub_count;
        layer_parameters(l->layers[i], &sub_params, &sub_count);
        total_params += sub_count;
        if (sub_params != NULL) {
            free(sub_params);
        }
    }

    if (total_params == 0) {
        *out_params = NULL;
        *out_count = 0;
        return;
    }

    Tensor **all_params = (Tensor **)malloc(total_params * sizeof(Tensor *));
    assert(all_params != NULL && "malloc failed");

    size_t current_idx = 0;
    for (size_t i = 0; i < l->count; ++i) {
        Tensor **sub_params;
        size_t sub_count;
        layer_parameters(l->layers[i], &sub_params, &sub_count);
        for (size_t j = 0; j < sub_count; ++j) {
            all_params[current_idx++] = sub_params[j];
        }
        if (sub_params != NULL) {
            free(sub_params);
        }
    }

    *out_params = all_params;
    *out_count = total_params;
}

Layer *layer_sequential_create(Layer **layers, size_t count) {
    SequentialLayer *l = (SequentialLayer *)calloc(1, sizeof(SequentialLayer));
    assert(l != NULL && "calloc failed");

    l->base.forward = sequential_forward;
    l->base.free = sequential_free;
    l->base.parameters = sequential_parameters;
    l->base.name = "Sequential";

    l->layers = (Layer **)malloc(count * sizeof(Layer *));
    assert(l->layers != NULL && "malloc failed");
    if (layers != NULL && count > 0) {
        memcpy(l->layers, layers, count * sizeof(Layer *));
    }
    l->count = count;

    return (Layer *)l;
}
