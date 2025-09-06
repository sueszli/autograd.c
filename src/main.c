#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Type definitions
typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int32_t i32;
typedef int64_t i64;

// CIFAR-10 constants
#define CIFAR10_IMAGE_SIZE 32
#define CIFAR10_NUM_CHANNELS 3
#define CIFAR10_PIXELS_PER_IMAGE (CIFAR10_IMAGE_SIZE * CIFAR10_IMAGE_SIZE * CIFAR10_NUM_CHANNELS)
#define CIFAR10_RECORD_SIZE (1 + CIFAR10_PIXELS_PER_IMAGE)
#define CIFAR10_NUM_CLASSES 10

// CIFAR-10 data structures
typedef struct {
    u8 label;
    u8 data[CIFAR10_PIXELS_PER_IMAGE];
} cifar10_sample_t;

typedef struct {
    cifar10_sample_t *samples;
    u64 num_samples;
} cifar10_batch_t;

// Class names
const char *CLASS_NAMES[CIFAR10_NUM_CLASSES] = {
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
};

// Load CIFAR-10 batch from file
cifar10_batch_t *load_batch(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        return NULL;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    i64 file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Calculate number of samples
    u64 num_samples = (u64)file_size / CIFAR10_RECORD_SIZE;
    if (file_size % CIFAR10_RECORD_SIZE != 0) {
        fclose(file);
        return NULL;
    }

    // Allocate batch
    cifar10_batch_t *batch = malloc(sizeof(cifar10_batch_t));
    if (!batch) {
        fclose(file);
        return NULL;
    }

    // Allocate samples
    batch->samples = malloc(num_samples * sizeof(cifar10_sample_t));
    if (!batch->samples) {
        free(batch);
        fclose(file);
        return NULL;
    }

    batch->num_samples = num_samples;

    // Read samples
    for (u64 i = 0; i < num_samples; i++) {
        if (fread(&batch->samples[i].label, 1, 1, file) != 1) {
            free(batch->samples);
            free(batch);
            fclose(file);
            return NULL;
        }

        if (fread(batch->samples[i].data, 1, CIFAR10_PIXELS_PER_IMAGE, file) != CIFAR10_PIXELS_PER_IMAGE) {
            free(batch->samples);
            free(batch);
            fclose(file);
            return NULL;
        }
    }

    fclose(file);
    return batch;
}

// Free batch memory
void free_batch(cifar10_batch_t *batch) {
    if (batch) {
        free(batch->samples);
        free(batch);
    }
}

// Print sample information
void print_sample_info(const cifar10_sample_t *sample) {
    if (sample && sample->label < CIFAR10_NUM_CLASSES) {
        printf("Label: %d (%s)\n", sample->label, CLASS_NAMES[sample->label]);
    }
}

int main(void) {
    printf("Loading CIFAR-10 data...\n");

    cifar10_batch_t *batch = load_batch("/workspace/data/data_batch_1.bin");
    if (!batch) {
        printf("Failed to load CIFAR-10 batch\n");
        return EXIT_FAILURE;
    }

    printf("Loaded %zu samples\n", batch->num_samples);

    for (int i = 0; i < 5 && i < (int)batch->num_samples; i++) {
        printf("\nSample %d:\n", i);
        print_sample_info(&batch->samples[i]);

        printf("First few RGB values: ");
        for (int j = 0; j < 9; j++) {
            printf("%d ", batch->samples[i].data[j]);
        }
        printf("...\n");
    }

    free_batch(batch);
    return EXIT_SUCCESS;
}