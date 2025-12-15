#pragma once

#include "tensor.h"
#include <stdbool.h>

// layer base type using vtable polymorphism
typedef struct Layer Layer;
typedef Tensor *(*LayerForwardFunc)(Layer *layer, const Tensor *input, bool training);
typedef void (*LayerFreeFunc)(Layer *layer);
typedef void (*LayerParametersFunc)(Layer *layer, Tensor ***out_params, size_t *out_count);

struct Layer {
    LayerForwardFunc forward;       // forward pass
    LayerFreeFunc free;             // cleanup
    LayerParametersFunc parameters; // get trainable params
    char *name;                     // for debugging
};

// forward pass through layer
Tensor *layer_forward(Layer *layer, const Tensor *input, bool training);

// frees layer resources
void layer_free(Layer *layer);

// gets trainable parameters
void layer_parameters(Layer *layer, Tensor ***out_params, size_t *out_count);

//
// layer constructors
//

// linear: y = xW + b
Layer *layer_linear_create(uint64_t in_features, uint64_t features_out, bool bias);

// dropout: randomly zeros elements during training
Layer *layer_dropout_create(float32_t p);

// sequential: chains layers together
Layer *layer_sequential_create(Layer **layers, size_t count);
