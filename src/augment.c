#include "augment.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

// applies a random horizontal flip to the tensor with probability p.
// only makes sense for images where semantics are invariant to horizontal flipping.
void random_horizontal_flip(Tensor *t, float32_t p) {
    assert(t != NULL);
    assert(t->data != NULL);
    assert(p >= 0.0f && p <= 1.0f);

    float32_t rand_float = (float32_t)rand() / (float32_t)RAND_MAX;
    if (rand_float >= p) {
        return;
    }

    uint64_t width = t->shape[t->ndim - 1];
    assert(width > 0);
    assert(t->strides[t->ndim - 1] == 1 && "tensor must be contiguous in last dimension");

    Tensor *out = tensor_create(NULL, t->shape, t->ndim, t->requires_grad);
    assert(out != NULL);

    uint64_t num_rows = t->size / width;
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
    assert(t->ndim == 2 || t->ndim == 3);

    // 2D case: [H, W]
    if (t->ndim == 2) {
        uint64_t h = t->shape[0];
        uint64_t w = t->shape[1];
        uint64_t padded_h = h + 2 * padding;
        uint64_t padded_w = w + 2 * padding;
        uint64_t max_top = padded_h >= target_h ? padded_h - target_h : 0;
        uint64_t max_left = padded_w >= target_w ? padded_w - target_w : 0;
        uint64_t top = max_top > 0 ? ((uint64_t)rand() % (max_top + 1)) : 0;
        uint64_t left = max_left > 0 ? ((uint64_t)rand() % (max_left + 1)) : 0;

        const uint64_t out_shape[2] = {target_h, target_w};
        Tensor *out = tensor_zeros(out_shape, 2, t->requires_grad);
        assert(out != NULL);

        for (uint64_t oy = 0; oy < target_h; oy++) {
            for (uint64_t ox = 0; ox < target_w; ox++) {
                int64_t iy = (int64_t)(top + oy) - (int64_t)padding;
                int64_t ix = (int64_t)(left + ox) - (int64_t)padding;
                if (iy < 0 || iy >= (int64_t)h || ix < 0 || ix >= (int64_t)w) {
                    continue;
                }
                uint64_t out_idx = oy * out->strides[0] + ox * out->strides[1];
                uint64_t in_idx = (uint64_t)iy * t->strides[0] + (uint64_t)ix * t->strides[1];
                out->data[out_idx] = t->data[in_idx];
            }
        }

        free(t->data);
        free(t->shape);
        free(t->strides);
        t->data = out->data;
        t->shape = out->shape;
        t->strides = out->strides;
        t->size = out->size;
        out->data = NULL;
        out->shape = NULL;
        out->strides = NULL;
        tensor_free(out);
        return;
    }

    // 3D case: detect [C, H, W] vs [H, W, C]
    bool is_chw = t->shape[0] <= 4;
    uint64_t h = is_chw ? t->shape[1] : t->shape[0];
    uint64_t w = is_chw ? t->shape[2] : t->shape[1];
    uint64_t c = is_chw ? t->shape[0] : t->shape[2];

    uint64_t padded_h = h + 2 * padding;
    uint64_t padded_w = w + 2 * padding;
    uint64_t max_top = padded_h >= target_h ? padded_h - target_h : 0;
    uint64_t max_left = padded_w >= target_w ? padded_w - target_w : 0;
    uint64_t top = max_top > 0 ? ((uint64_t)rand() % (max_top + 1)) : 0;
    uint64_t left = max_left > 0 ? ((uint64_t)rand() % (max_left + 1)) : 0;

    uint64_t out_shape[3];
    if (is_chw) {
        out_shape[0] = c;
        out_shape[1] = target_h;
        out_shape[2] = target_w;
    } else {
        out_shape[0] = target_h;
        out_shape[1] = target_w;
        out_shape[2] = c;
    }

    Tensor *out = tensor_zeros(out_shape, 3, t->requires_grad);
    assert(out != NULL);

    for (uint64_t oy = 0; oy < target_h; oy++) {
        for (uint64_t ox = 0; ox < target_w; ox++) {
            int64_t iy = (int64_t)(top + oy) - (int64_t)padding;
            int64_t ix = (int64_t)(left + ox) - (int64_t)padding;
            if (iy < 0 || iy >= (int64_t)h || ix < 0 || ix >= (int64_t)w) {
                continue;
            }

            for (uint64_t ci = 0; ci < c; ci++) {
                uint64_t out_idx, in_idx;
                if (is_chw) {
                    out_idx = ci * out->strides[0] + oy * out->strides[1] + ox * out->strides[2];
                    in_idx = ci * t->strides[0] + (uint64_t)iy * t->strides[1] + (uint64_t)ix * t->strides[2];
                } else {
                    out_idx = oy * out->strides[0] + ox * out->strides[1] + ci * out->strides[2];
                    in_idx = (uint64_t)iy * t->strides[0] + (uint64_t)ix * t->strides[1] + ci * t->strides[2];
                }
                out->data[out_idx] = t->data[in_idx];
            }
        }
    }

    free(t->data);
    free(t->shape);
    free(t->strides);
    t->data = out->data;
    t->shape = out->shape;
    t->strides = out->strides;
    t->size = out->size;
    t->ndim = out->ndim;
    out->data = NULL;
    out->shape = NULL;
    out->strides = NULL;
    tensor_free(out);
}
