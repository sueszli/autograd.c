#pragma once

#include "tensor.h"
#include <stdbool.h>

//
// layer type definitions
//

typedef struct Layer Layer;

// function pointer types for polymorphism aka "vtable"
typedef Tensor *(*LayerForwardFunc)(Layer *layer, const Tensor *input, bool training);
typedef void (*LayerFreeFunc)(Layer *layer);
typedef void (*LayerParametersFunc)(Layer *layer, Tensor ***out_params, size_t *out_count);

struct Layer {
    LayerForwardFunc forward;
    LayerFreeFunc free;
    LayerParametersFunc parameters;
    char *name;
};

//
// public api constructors
//

/**
 * creates a linear (dense) layer.
 * y = xW + b
 *
 * @param in_features  size of each input sample
 * @param features_out size of each output sample
 * @param bias         if true, learns an additive bias
 * @return             pointer to new layer
 */
Layer *layer_linear_create(uint64_t in_features, uint64_t features_out, bool bias);

/**
 * creates a dropout layer.
 * during training: randomly zeroes elements with probability p.
 * during inference: identity.
 *
 * @param p            probability of an element to be zeroed. (0 <= p <= 1)
 * @return             pointer to new layer
 */
Layer *layer_dropout_create(float32_t p);

/**
 * creates a sequential container layer.
 * chains layers together: y = layerN(...layer2(layer1(x))...)
 *
 * @param layers       array of layer pointers. takes ownership of these pointers.
 * @param count        number of layers
 * @return             pointer to new layer
 */
Layer *layer_sequential_create(Layer **layers, size_t count);

//
// public api methods
//

/**
 * forward pass through the layer.
 *
 * @param layer    layer instance
 * @param input    input tensor (const)
 * @param training true if training mode (affects dropout)
 * @return         output tensor (newly allocated)
 */
Tensor *layer_forward(Layer *layer, const Tensor *input, bool training);

/**
 * frees the layer and its internal resources (weights, biases, sub-layers).
 *
 * @param layer layer instance
 */
void layer_free(Layer *layer);

/**
 * retrieves trainable parameters (weights, biases).
 *
 * @param layer      layer instance
 * @param out_params output pointer to array of tensor pointers (caller must not free the tensors, but may need to free the array if it was dynamically allocated - in this api we will return a copy of pointers or reference internal list. let's specify: caller frees the array, not the tensors.)
 * @param out_count  output number of parameters
 */
void layer_parameters(Layer *layer, Tensor ***out_params, size_t *out_count);
