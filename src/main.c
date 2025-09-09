#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "datasets/cifar10.h"
#include "eval/metrics.h"
#include "tensor/autograd.h"
#include "tensor/tensor.h"
#include "utils/defer.h"
#include "utils/tqdm.h"
#include "utils/types.h"

#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001f
#define EPOCHS 2
#define BATCH_SIZE 32

typedef struct {
    Tensor **items;
    int capacity;
    int count;
} TensorList;

void list_add(TensorList *list, Tensor *t) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->items = realloc(list->items, list->capacity * sizeof(Tensor *));
    }
    list->items[list->count++] = t;
}

TensorList *list_create() {
    TensorList *list = malloc(sizeof(TensorList));
    list->capacity = 16;
    list->count = 0;
    list->items = malloc(list->capacity * sizeof(Tensor *));
    return list;
}

void list_destroy_tensors(TensorList *list) {
    for (int i = 0; i < list->count; i++) {
        tensor_destroy(list->items[i]);
    }
    free(list->items);
    free(list);
}

typedef struct {
    Tensor *weights;
    Tensor *bias;
} layer_t;

typedef struct {
    layer_t hidden_layer;
    layer_t output_layer;
} neural_network_t;

f32 random_weight(void) {
    static const f32 inv_rand_max = 2.0f / (f32)RAND_MAX;
    i32 r = rand();
    return (f32)r * inv_rand_max - 1.0f;
}

void init_layer(layer_t *layer, int input_size, int output_size) {
    int w_shape[] = {input_size, output_size};
    float *w_data = malloc((size_t)input_size * (size_t)output_size * sizeof(float));
    f32 xavier_multiplier = sqrtf(2.0f / (f32)(input_size + output_size)) * 0.1f;
    for (int i = 0; i < input_size * output_size; i++) {
        w_data[i] = random_weight() * xavier_multiplier;
    }
    layer->weights = tensor_create(w_data, w_shape, 2, true);
    free(w_data);

    int b_shape[] = {1, output_size};
    float *b_data = calloc(output_size, sizeof(float));
    layer->bias = tensor_create(b_data, b_shape, 2, true);
    free(b_data);
}

void free_layer(layer_t *layer) {
    tensor_destroy(layer->weights);
    tensor_destroy(layer->bias);
}

neural_network_t *create_network(void) {
    neural_network_t *network = malloc(sizeof(neural_network_t));
    assert(network != NULL);

    init_layer(&network->hidden_layer, INPUT_SIZE, HIDDEN_SIZE);
    init_layer(&network->output_layer, HIDDEN_SIZE, OUTPUT_SIZE);

    return network;
}

void free_network(neural_network_t *network) {
    if (network == NULL)
        return;
    free_layer(&network->hidden_layer);
    free_layer(&network->output_layer);
    free(network);
}

void normalize_input(u8 *input, float *output, u32 size) {
    const f32 inv_255 = 1.0f / 255.0f;
    for (u32 i = 0; i < size; i++) {
        output[i] = (f32)input[i] * inv_255;
    }
}

Tensor *forward_pass(neural_network_t *network, Tensor *input, TensorList *graph) {
    Tensor *hidden1 = tensor_matmul(input, network->hidden_layer.weights);
    list_add(graph, hidden1);
    Tensor *hidden2 = tensor_add(hidden1, network->hidden_layer.bias);
    list_add(graph, hidden2);
    Tensor *hidden3 = tensor_relu(hidden2);
    list_add(graph, hidden3);

    Tensor *output1 = tensor_matmul(hidden3, network->output_layer.weights);
    list_add(graph, output1);
    Tensor *output2 = tensor_add(output1, network->output_layer.bias);
    list_add(graph, output2);

    return output2;
}

cifar10_class_t predict(neural_network_t *network, sample_t *sample) {
    float *normalized_input = malloc(INPUT_SIZE * sizeof(float));
    normalize_input(sample->data, normalized_input, INPUT_SIZE);

    int input_shape[] = {1, INPUT_SIZE};
    Tensor *input_tensor = tensor_create(normalized_input, input_shape, 2, false);
    free(normalized_input);

    TensorList *graph = list_create();
    Tensor *output = forward_pass(network, input_tensor, graph);

    cifar10_class_t predicted = 0;
    f32 max_prob = output->data[0];
    for (cifar10_class_t i = 1; i < OUTPUT_SIZE; i++) {
        if (output->data[i] > max_prob) {
            max_prob = output->data[i];
            predicted = i;
        }
    }
    tensor_destroy(input_tensor);
    list_destroy_tensors(graph);
    return predicted;
}

f32 evaluate_accuracy(neural_network_t *network, sample_t *samples, u64 count) {
    cifar10_class_t *true_labels = malloc(count * sizeof(cifar10_class_t));
    cifar10_class_t *predicted_labels = malloc(count * sizeof(cifar10_class_t));
    assert(true_labels != NULL);
    assert(predicted_labels != NULL);

    u64 eval_step = count / 100;
    if (eval_step == 0) {
        eval_step = 1;
    }

    for (u64 i = 0; i < count; i++) {
        true_labels[i] = samples[i].label;
        predicted_labels[i] = predict(network, &samples[i]);

        if (i % eval_step == 0 || i == count - 1) {
            f32 current_accuracy = accuracy(true_labels, predicted_labels, i + 1);
            char info[64];
            snprintf(info, sizeof(info), "Accuracy: %.2f%%", current_accuracy * 100.0f);
            tqdm(i + 1, count, "Evaluating", info);
        }
    }

    f32 final_accuracy = accuracy(true_labels, predicted_labels, count);
    free(true_labels);
    free(predicted_labels);
    return final_accuracy;
}

void train_network(neural_network_t *network, sample_t *train_samples, sample_t *test_samples) {
    printf("Starting training...\n");
    u32 total_batches = (NUM_TRAIN_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE;
    float *normalized_input = malloc(INPUT_SIZE * sizeof(float));

    for (u32 epoch = 0; epoch < EPOCHS; epoch++) {
        printf("\nEpoch %u/%u:\n", epoch + 1, EPOCHS);
        f32 total_loss = 0.0f;
        u32 batch_count = 0;

        for (u64 i = 0; i < NUM_TRAIN_SAMPLES; i += BATCH_SIZE) {
            u64 batch_end = (i + BATCH_SIZE < NUM_TRAIN_SAMPLES) ? i + BATCH_SIZE : NUM_TRAIN_SAMPLES;

            tensor_zero_grad(network->hidden_layer.weights);
            tensor_zero_grad(network->hidden_layer.bias);
            tensor_zero_grad(network->output_layer.weights);
            tensor_zero_grad(network->output_layer.bias);

            f32 batch_loss = 0.0f;
            TensorList *graph = list_create();
            u32 actual_batch_size = batch_end - i;

            for (u64 j = i; j < batch_end; j++) {
                normalize_input(train_samples[j].data, normalized_input, INPUT_SIZE);
                int input_shape[] = {1, INPUT_SIZE};
                Tensor *input_tensor = tensor_create(normalized_input, input_shape, 2, false);

                Tensor *output = forward_pass(network, input_tensor, graph);
                Tensor *loss = tensor_cross_entropy(output, train_samples[j].label);
                batch_loss += loss->data[0];

                tensor_backward(loss);

                tensor_destroy(input_tensor);
                tensor_destroy(loss);
            }

            // Update weights with averaged gradients
            f32 lr_over_batch = LEARNING_RATE / (f32)actual_batch_size;
            for (size_t k = 0; k < tensor_size(network->hidden_layer.weights); ++k)
                network->hidden_layer.weights->data[k] -= lr_over_batch * network->hidden_layer.weights->grad->data[k];
            for (size_t k = 0; k < tensor_size(network->hidden_layer.bias); ++k)
                network->hidden_layer.bias->data[k] -= lr_over_batch * network->hidden_layer.bias->grad->data[k];
            for (size_t k = 0; k < tensor_size(network->output_layer.weights); ++k)
                network->output_layer.weights->data[k] -= lr_over_batch * network->output_layer.weights->grad->data[k];
            for (size_t k = 0; k < tensor_size(network->output_layer.bias); ++k)
                network->output_layer.bias->data[k] -= lr_over_batch * network->output_layer.bias->grad->data[k];

            list_destroy_tensors(graph);

            batch_loss /= (f32)actual_batch_size;
            total_loss += batch_loss;
            batch_count++;

            char info[128];
            snprintf(info, sizeof(info), "Batch %u/%u | Loss: %.4f", batch_count, total_batches, batch_loss);
            tqdm(batch_count, total_batches, "Training", info);
        }

        printf("\n\nEvaluating epoch %u...\n", epoch + 1);
        f32 avg_loss = total_loss / (f32)batch_count;
        f32 train_accuracy = evaluate_accuracy(network, train_samples, 1000); // evaluate on a subset for speed
        f32 test_accuracy = evaluate_accuracy(network, test_samples, NUM_TEST_SAMPLES);

        printf("✓ Epoch %u Complete: Loss=%.4f, Train Acc=%.4f (%.2f%%), Test Acc=%.4f (%.2f%%)\n", epoch + 1, avg_loss, train_accuracy, train_accuracy * 100.0f, test_accuracy, test_accuracy * 100.0f);
    }
    free(normalized_input);
}

i32 main(void) {
    srand((u32)time(NULL));

    static train_samples_t train_data;
    static test_samples_t test_data;
    load_train_samples_to_buffer(train_data);
    load_test_samples_to_buffer(test_data);

    printf("Network architecture: %d -> %d -> %d\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    neural_network_t *network = create_network();
    defer({ free_network(network); });

    train_network(network, train_data, test_data);

    f32 final_accuracy = evaluate_accuracy(network, test_data, NUM_TEST_SAMPLES);
    printf("Final test accuracy: %.4f (%.2f%%)\n", final_accuracy, final_accuracy * 100.0f);

    return EXIT_SUCCESS;
}
