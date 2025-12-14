#pragma once

#include "tensor.h"
#include <stdbool.h>

// ----------------------------------------------------------------------------
// Layer Type Definitions
// ----------------------------------------------------------------------------

typedef struct Layer Layer;

// Function pointer types for polymorphism aka "vtable"
typedef Tensor *(*LayerForwardFunc)(Layer *layer, const Tensor *input, bool training);
typedef void (*LayerFreeFunc)(Layer *layer);
typedef void (*LayerParametersFunc)(Layer *layer, Tensor ***out_params, size_t *out_count);

struct Layer {
    LayerForwardFunc forward;
    LayerFreeFunc free;
    LayerParametersFunc parameters;
    char *name;
};

// ----------------------------------------------------------------------------
// Public API Constructors
// ----------------------------------------------------------------------------

/**
 * Creates a Linear (Dense) layer.
 * y = xW + b
 *
 * @param in_features  Size of each input sample
 * @param out_features Size of each output sample
 * @param bias         If true, learns an additive bias
 * @return             Pointer to new Layer
 */
Layer *layer_linear_create(uint64_t in_features, uint64_t features_out, bool bias);

/**
 * Creates a Dropout layer.
 * During training: randomly zeroes elements with probability p.
 * During inference: identity.
 *
 * @param p            Probability of an element to be zeroed. (0 <= p <= 1)
 * @return             Pointer to new Layer
 */
Layer *layer_dropout_create(float32_t p);

/**
 * Creates a Sequential container layer.
 * Chains layers together: y = layerN(...layer2(layer1(x))...)
 *
 * @param layers       Array of Layer pointers. Takes ownership of these pointers.
 * @param count        Number of layers
 * @return             Pointer to new Layer
 */
Layer *layer_sequential_create(Layer **layers, size_t count);

// ----------------------------------------------------------------------------
// Public API Methods
// ----------------------------------------------------------------------------

/**
 * Forward pass through the layer.
 *
 * @param layer    Layer instance
 * @param input    Input tensor (const)
 * @param training True if training mode (affects Dropout)
 * @return         Output tensor (newly allocated)
 */
Tensor *layer_forward(Layer *layer, const Tensor *input, bool training);

/**
 * Frees the layer and its internal resources (weights, biases, sub-layers).
 *
 * @param layer Layer instance
 */
void layer_free(Layer *layer);

/**
 * Retrieves trainable parameters (weights, biases).
 *
 * @param layer      Layer instance
 * @param out_params Output pointer to array of Tensor pointers (caller must not free the tensors, but may need to free the array if it was dynamically allocated - in this API we will return a copy of pointers or reference internal list. Let's specify: caller frees the array, not the Tensors.)
 * @param out_count  Output number of parameters
 */
void layer_parameters(Layer *layer, Tensor ***out_params, size_t *out_count);
