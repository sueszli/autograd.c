#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "datasets/cifar10.h"
#include "utils/defer.h"
#include "utils/types.h"

#define INPUT_SIZE 3072 // 32*32*3
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001f
#define EPOCHS 2
#define BATCH_SIZE 100

typedef struct {
    f32 *weights;
    f32 *bias;
    u32 input_size;
    u32 output_size;
} layer_t;

typedef struct {
    layer_t hidden_layer;
    layer_t output_layer;
    f32 *hidden_output;
    f32 *final_output;
    f32 *hidden_delta;
    f32 *output_delta;
} neural_network_t;

f32 simple_sqrt(f32 x) {
    if (x <= 0.0f)
        return 0.0f;
    f32 guess = x;
    for (int i = 0; i < 10; i++) {
        guess = (guess + x / guess) * 0.5f;
    }
    return guess;
}

f32 simple_exp(f32 x) {
    if (x > 88.0f)
        return 1e38f;
    if (x < -88.0f)
        return 0.0f;

    f32 result = 1.0f;
    f32 term = 1.0f;
    for (int i = 1; i < 20; i++) {
        term *= x / (f32)i;
        result += term;
        if (term < 1e-10f)
            break;
    }
    return result;
}

f32 simple_log(f32 x) {
    if (x <= 0.0f)
        return -1e38f;
    if (x == 1.0f)
        return 0.0f;

    x = (x - 1.0f) / (x + 1.0f);
    f32 x2 = x * x;
    f32 result = x;
    f32 term = x;

    for (int i = 1; i < 20; i++) {
        term *= x2;
        result += term / (f32)(2 * i + 1);
        if (term < 1e-10f)
            break;
    }
    return 2.0f * result;
}

f32 random_weight(void) {
    int r = rand();
    return ((f32)r / (f32)RAND_MAX) * 2.0f - 1.0f;
}

void init_layer(layer_t *layer, u32 input_size, u32 output_size) {
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->weights = malloc(input_size * output_size * sizeof(f32));
    layer->bias = calloc(output_size, sizeof(f32));

    assert(layer->weights != NULL);
    assert(layer->bias != NULL);

    f32 xavier_std = simple_sqrt(2.0f / (f32)(input_size + output_size));
    for (u32 i = 0; i < input_size * output_size; i++) {
        layer->weights[i] = random_weight() * xavier_std * 0.1f;
    }
}

void free_layer(layer_t *layer) {
    free(layer->weights);
    free(layer->bias);
}

neural_network_t *create_network(void) {
    neural_network_t *network = malloc(sizeof(neural_network_t));
    assert(network != NULL);

    init_layer(&network->hidden_layer, INPUT_SIZE, HIDDEN_SIZE);
    init_layer(&network->output_layer, HIDDEN_SIZE, OUTPUT_SIZE);

    network->hidden_output = malloc(HIDDEN_SIZE * sizeof(f32));
    network->final_output = malloc(OUTPUT_SIZE * sizeof(f32));
    network->hidden_delta = malloc(HIDDEN_SIZE * sizeof(f32));
    network->output_delta = malloc(OUTPUT_SIZE * sizeof(f32));

    assert(network->hidden_output != NULL);
    assert(network->final_output != NULL);
    assert(network->hidden_delta != NULL);
    assert(network->output_delta != NULL);

    return network;
}

void free_network(neural_network_t *network) {
    if (network == NULL)
        return;

    free_layer(&network->hidden_layer);
    free_layer(&network->output_layer);
    free(network->hidden_output);
    free(network->final_output);
    free(network->hidden_delta);
    free(network->output_delta);
    free(network);
}

f32 relu(f32 x) { return x > 0.0f ? x : 0.0f; }

f32 relu_derivative(f32 x) { return x > 0.0f ? 1.0f : 0.0f; }

void softmax(f32 *input, f32 *output, u32 size) {
    f32 max_val = input[0];
    for (u32 i = 1; i < size; i++) {
        if (input[i] > max_val)
            max_val = input[i];
    }

    f32 sum = 0.0f;
    for (u32 i = 0; i < size; i++) {
        output[i] = simple_exp(input[i] - max_val);
        sum += output[i];
    }

    for (u32 i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

void normalize_input(u8 *input, f32 *output, u32 size) {
    for (u32 i = 0; i < size; i++) {
        output[i] = (f32)input[i] / 255.0f;
    }
}

void forward_pass(neural_network_t *network, f32 *input) {
    layer_t *hidden = &network->hidden_layer;
    layer_t *output = &network->output_layer;

    for (u32 j = 0; j < HIDDEN_SIZE; j++) {
        f32 sum = hidden->bias[j];
        for (u32 i = 0; i < INPUT_SIZE; i++) {
            sum += input[i] * hidden->weights[i * HIDDEN_SIZE + j];
        }
        network->hidden_output[j] = relu(sum);
    }

    for (u32 j = 0; j < OUTPUT_SIZE; j++) {
        f32 sum = output->bias[j];
        for (u32 i = 0; i < HIDDEN_SIZE; i++) {
            sum += network->hidden_output[i] * output->weights[i * OUTPUT_SIZE + j];
        }
        network->final_output[j] = sum;
    }

    softmax(network->final_output, network->final_output, OUTPUT_SIZE);
}

f32 compute_loss(neural_network_t *network, u8 target_label) {
    f32 prob = network->final_output[target_label];
    if (prob < 1e-7f)
        prob = 1e-7f;
    if (prob > 1.0f - 1e-7f)
        prob = 1.0f - 1e-7f;
    return -simple_log(prob);
}

f32 clip_gradient(f32 grad, f32 max_norm) {
    if (grad > max_norm)
        return max_norm;
    if (grad < -max_norm)
        return -max_norm;
    return grad;
}

void backward_pass(neural_network_t *network, f32 *input, u8 target_label) {
    layer_t *hidden = &network->hidden_layer;
    layer_t *output = &network->output_layer;

    for (u32 i = 0; i < OUTPUT_SIZE; i++) {
        network->output_delta[i] = network->final_output[i];
        if (i == target_label) {
            network->output_delta[i] -= 1.0f;
        }
        network->output_delta[i] = clip_gradient(network->output_delta[i], 5.0f);
    }

    for (u32 i = 0; i < HIDDEN_SIZE; i++) {
        f32 sum = 0.0f;
        for (u32 j = 0; j < OUTPUT_SIZE; j++) {
            sum += network->output_delta[j] * output->weights[i * OUTPUT_SIZE + j];
        }
        network->hidden_delta[i] = clip_gradient(sum * relu_derivative(network->hidden_output[i]), 5.0f);
    }

    for (u32 i = 0; i < HIDDEN_SIZE; i++) {
        for (u32 j = 0; j < OUTPUT_SIZE; j++) {
            f32 grad = clip_gradient(network->output_delta[j] * network->hidden_output[i], 1.0f);
            output->weights[i * OUTPUT_SIZE + j] -= LEARNING_RATE * grad;
        }
    }
    for (u32 j = 0; j < OUTPUT_SIZE; j++) {
        f32 grad = clip_gradient(network->output_delta[j], 1.0f);
        output->bias[j] -= LEARNING_RATE * grad;
    }

    for (u32 i = 0; i < INPUT_SIZE; i++) {
        for (u32 j = 0; j < HIDDEN_SIZE; j++) {
            f32 grad = clip_gradient(network->hidden_delta[j] * input[i], 1.0f);
            hidden->weights[i * HIDDEN_SIZE + j] -= LEARNING_RATE * grad;
        }
    }
    for (u32 j = 0; j < HIDDEN_SIZE; j++) {
        f32 grad = clip_gradient(network->hidden_delta[j], 1.0f);
        hidden->bias[j] -= LEARNING_RATE * grad;
    }
}

u8 predict(neural_network_t *network, sample_t *sample) {
    f32 input[INPUT_SIZE];
    normalize_input(sample->data, input, INPUT_SIZE);
    forward_pass(network, input);

    u8 predicted = 0;
    f32 max_prob = network->final_output[0];
    for (u8 i = 1; i < OUTPUT_SIZE; i++) {
        if (network->final_output[i] > max_prob) {
            max_prob = network->final_output[i];
            predicted = i;
        }
    }
    return predicted;
}

f32 evaluate_accuracy(neural_network_t *network, sample_arr_t *test_samples) {
    u32 correct = 0;

    for (u64 i = 0; i < test_samples->count; i++) {
        u8 predicted = predict(network, &test_samples->samples[i]);
        if (predicted == test_samples->samples[i].label) {
            correct++;
        }
    }

    return (f32)correct / (f32)test_samples->count;
}

void print_progress_bar(u32 current, u32 total, u32 width) {
    f32 progress = (f32)current / (f32)total;
    u32 filled = (u32)(progress * (f32)width);

    printf("\r[");
    for (u32 i = 0; i < width; i++) {
        if (i < filled)
            printf("█");
        else
            printf(" ");
    }
    printf("] %3.0f%%", progress * 100.0f);
    fflush(stdout);
}

void train_network(neural_network_t *network, sample_arr_t *train_samples, sample_arr_t *test_samples) {
    printf("Starting training...\n");
    u32 total_batches = (u32)((train_samples->count + BATCH_SIZE - 1) / BATCH_SIZE);

    for (u32 epoch = 0; epoch < EPOCHS; epoch++) {
        printf("\nEpoch %u/%u:\n", epoch + 1, EPOCHS);
        f32 total_loss = 0.0f;
        u32 batch_count = 0;

        for (u64 i = 0; i < train_samples->count; i += BATCH_SIZE) {
            f32 batch_loss = 0.0f;
            u64 batch_end = (i + BATCH_SIZE < train_samples->count) ? i + BATCH_SIZE : train_samples->count;

            for (u64 j = i; j < batch_end; j++) {
                f32 input[INPUT_SIZE];
                normalize_input(train_samples->samples[j].data, input, INPUT_SIZE);

                forward_pass(network, input);
                batch_loss += compute_loss(network, train_samples->samples[j].label);
                backward_pass(network, input, train_samples->samples[j].label);
            }

            batch_loss /= (f32)(batch_end - i);
            total_loss += batch_loss;
            batch_count++;

            print_progress_bar(batch_count, total_batches, 30);
            printf(" Batch %u/%u | Loss: %.4f", batch_count, total_batches, batch_loss);

            if (batch_count % 50 == 0 || batch_count == total_batches) {
                f32 current_avg_loss = total_loss / (f32)batch_count;
                printf(" | Avg Loss: %.4f", current_avg_loss);
            }
        }

        printf("\n\nEvaluating epoch %u...\n", epoch + 1);
        f32 avg_loss = total_loss / (f32)batch_count;
        f32 train_accuracy = evaluate_accuracy(network, train_samples);
        f32 test_accuracy = evaluate_accuracy(network, test_samples);

        printf("✓ Epoch %u Complete: Loss=%.4f, Train Acc=%.4f (%.2f%%), Test Acc=%.4f (%.2f%%)\n", epoch + 1, avg_loss, train_accuracy, train_accuracy * 100.0f, test_accuracy, test_accuracy * 100.0f);
    }
}

int main(void) {
    srand((u32)time(NULL));

    printf("Loading CIFAR-10 dataset...\n");
    sample_arr_t train_samples = get_train_samples();
    sample_arr_t test_samples = get_test_samples();
    assert(train_samples.samples != NULL);
    assert(test_samples.samples != NULL);

    printf("Loaded %lu training samples and %lu test samples.\n", train_samples.count, test_samples.count);

    printf("Creating neural network...\n");
    neural_network_t *network = create_network();
    defer({ free_network(network); });

    printf("Network architecture: %d -> %d -> %d\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    train_network(network, &train_samples, &test_samples);

    printf("\nFinal evaluation:\n");
    f32 final_accuracy = evaluate_accuracy(network, &test_samples);
    printf("Final test accuracy: %.4f (%.2f%%)\n", final_accuracy, final_accuracy * 100.0f);

    printf("\nSample predictions:\n");
    for (u32 i = 0; i < 5; i++) {
        u8 predicted = predict(network, &test_samples.samples[i]);
        u8 actual = test_samples.samples[i].label;
        printf("Sample %u: Predicted=%s, Actual=%s %s\n", i + 1, get_class_name(predicted), get_class_name(actual), predicted == actual ? "✓" : "✗");
    }

    return EXIT_SUCCESS;
}
