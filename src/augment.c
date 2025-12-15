#include "augment.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static float32_t random_float() { return (float32_t)rand() / (float32_t)RAND_MAX; }

static uint64_t random_int_range(uint64_t min, uint64_t max) {
    if (min >= max) {
        return min;
    }
    return min + ((uint64_t)rand() % (max - min + 1));
}

/*
 * applies a random horizontal flip to the tensor with probability p.
 * only makes sense for images where semantics are invariant to horizontal flipping.
 */
void random_horizontal_flip(Tensor *t, float32_t p) {
    assert(t != NULL);
    assert(t->data != NULL);
    assert(p >= 0.0f && p <= 1.0f);

    if (random_float() >= p) {
        return;
    }

    Tensor *out = tensor_create(NULL, t->shape, t->ndim, t->requires_grad);
    assert(out != NULL);

    // flip width dimension (always last)
    uint64_t last_dim = t->ndim - 1;
    uint64_t width = t->shape[last_dim];
    assert(width > 0);
    uint64_t num_rows = t->size / width;

    assert(t->strides[last_dim] == 1 && "tensor must be contiguous in last dimension");

    // standard copy (can be vectorized)
    for (uint64_t r = 0; r < num_rows; r++) {
        const float32_t *src_row = t->data + r * width;
        float32_t *dst_row = out->data + r * width;
        for (uint64_t c = 0; c < width; c++) {
            dst_row[c] = src_row[width - 1 - c];
        }
    }

    free(t->data);
    t->data = out->data;
    out->data = NULL;
    tensor_free(out);
}

/*
 * applies a random crop to the tensor.
 * virtually pads the image with zeros, then selects a random window.
 *
 * example (padding=1, target=2x2):
 *
 *   input (2x2):     virtual padded (4x4):    random crop (2x2):
 *   [1, 1]           0  0  0  0               [0, 0]
 *   [1, 1]      ->   0 [1, 1] 0        ->     [0, 1]
 *                    0 [1, 1] 0               (if top=0, left=0)
 *                    0  0  0  0
 *
 * simulates translation invariance by shifting content.
 */
void random_crop(Tensor *t, uint64_t target_h, uint64_t target_w, uint64_t padding) {
    assert(t != NULL);
    assert(t->data != NULL);
    assert(target_h > 0);
    assert(target_w > 0);

    // identify layout
    uint64_t h, w, c_dim;
    bool is_chw = false;
    bool is_hwc = false;

    if (t->ndim == 2) {
        h = t->shape[0];
        w = t->shape[1];
    } else if (t->ndim == 3) {
        if (t->shape[0] <= 4) { // heuristic for C, H, W
            is_chw = true;
            c_dim = t->shape[0];
            h = t->shape[1];
            w = t->shape[2];
        } else { // heuristic for H, W, C
            is_hwc = true;
            h = t->shape[0];
            w = t->shape[1];
            c_dim = t->shape[2];
        }
    } else {
        assert(false && "RandomCrop only supports 2D or 3D tensors");
    }

    // calculate crop coordinates
    uint64_t padded_h = h + 2 * padding;
    uint64_t padded_w = w + 2 * padding;

    uint64_t max_top = padded_h >= target_h ? padded_h - target_h : 0;
    uint64_t max_left = padded_w >= target_w ? padded_w - target_w : 0;

    uint64_t top = random_int_range(0, max_top);
    uint64_t left = random_int_range(0, max_left);

    // prepare output shape
    uint64_t out_shape[3];
    if (t->ndim == 2) {
        out_shape[0] = target_h;
        out_shape[1] = target_w;
    } else if (is_chw) {
        out_shape[0] = c_dim;
        out_shape[1] = target_h;
        out_shape[2] = target_w;
    } else {
        out_shape[0] = target_h;
        out_shape[1] = target_w;
        out_shape[2] = c_dim;
    }

    Tensor *out = tensor_zeros(out_shape, t->ndim, t->requires_grad);
    assert(out != NULL);

    // processing: specialized loops to avoid branching inside hot loop
    if (t->ndim == 2) {
        for (uint64_t oy = 0; oy < target_h; oy++) {
            for (uint64_t ox = 0; ox < target_w; ox++) {
                int64_t iy = (int64_t)(top + oy) - (int64_t)padding;
                int64_t ix = (int64_t)(left + ox) - (int64_t)padding;

                if (iy >= 0 && iy < (int64_t)h && ix >= 0 && ix < (int64_t)w) {
                    uint64_t out_idx = oy * out->strides[0] + ox * out->strides[1];
                    uint64_t in_idx = (uint64_t)iy * t->strides[0] + (uint64_t)ix * t->strides[1];
                    out->data[out_idx] = t->data[in_idx];
                }
            }
        }
    } else if (is_chw) {
        for (uint64_t oy = 0; oy < target_h; oy++) {
            for (uint64_t ox = 0; ox < target_w; ox++) {
                int64_t iy = (int64_t)(top + oy) - (int64_t)padding;
                int64_t ix = (int64_t)(left + ox) - (int64_t)padding;

                if (iy >= 0 && iy < (int64_t)h && ix >= 0 && ix < (int64_t)w) {
                    for (uint64_t c = 0; c < c_dim; c++) {
                        uint64_t out_idx = c * out->strides[0] + oy * out->strides[1] + ox * out->strides[2];
                        uint64_t in_idx = c * t->strides[0] + (uint64_t)iy * t->strides[1] + (uint64_t)ix * t->strides[2];
                        out->data[out_idx] = t->data[in_idx];
                    }
                }
            }
        }
    } else if (is_hwc) {
        for (uint64_t oy = 0; oy < target_h; oy++) {
            for (uint64_t ox = 0; ox < target_w; ox++) {
                int64_t iy = (int64_t)(top + oy) - (int64_t)padding;
                int64_t ix = (int64_t)(left + ox) - (int64_t)padding;

                if (iy >= 0 && iy < (int64_t)h && ix >= 0 && ix < (int64_t)w) {
                    for (uint64_t c = 0; c < c_dim; c++) {
                        uint64_t out_idx = oy * out->strides[0] + ox * out->strides[1] + c * out->strides[2];
                        uint64_t in_idx = (uint64_t)iy * t->strides[0] + (uint64_t)ix * t->strides[1] + c * t->strides[2];
                        out->data[out_idx] = t->data[in_idx];
                    }
                }
            }
        }
    }

    // replace data
    free(t->data);
    free(t->shape);
    free(t->strides);

    t->data = out->data;
    t->shape = out->shape;
    t->strides = out->strides;
    t->size = out->size;
    t->ndim = out->ndim;

    // cleanup out container
    out->data = NULL;
    out->shape = NULL;
    out->strides = NULL;
    tensor_free(out);
}
