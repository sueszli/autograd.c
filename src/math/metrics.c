#include "../datasets/cifar10.h"
#include "../utils/types.h"
#include <math.h>
#include <stdio.h>

f32 accuracy(const u8 *true_labels, const u8 *predicted_labels, u64 length) {
    if (length == 0) {
        return 0.0f;
    }

    u64 correct = 0;
    for (u64 i = 0; i < length; i++) {
        if (true_labels[i] == predicted_labels[i]) {
            correct++;
        }
    }

    return (f32)correct / (f32)length;
}

f32 precision(const u8 *true_labels, const u8 *predicted_labels, u64 length, u8 class_id) {
    if (length == 0) {
        return 0.0f;
    }

    u64 true_positives = 0;
    u64 predicted_positives = 0;

    for (u64 i = 0; i < length; i++) {
        if (predicted_labels[i] == class_id) {
            predicted_positives++;
            if (true_labels[i] == class_id) {
                true_positives++;
            }
        }
    }

    if (predicted_positives == 0) {
        return 0.0f;
    }
    return (f32)true_positives / (f32)predicted_positives;
}

f32 recall(const u8 *true_labels, const u8 *predicted_labels, u64 length, u8 class_id) {
    if (length == 0) {
        return 0.0f;
    }

    u64 true_positives = 0;
    u64 actual_positives = 0;

    for (u64 i = 0; i < length; i++) {
        if (true_labels[i] == class_id) {
            actual_positives++;
            if (predicted_labels[i] == class_id) {
                true_positives++;
            }
        }
    }

    if (actual_positives == 0) {
        return 0.0f;
    }
    return (f32)true_positives / (f32)actual_positives;
}

f32 f1_score(const u8 *true_labels, const u8 *predicted_labels, u64 length, u8 class_id) {
    f32 prec = precision(true_labels, predicted_labels, length, class_id);
    f32 rec = recall(true_labels, predicted_labels, length, class_id);

    if (prec + rec == 0.0f) {
        return 0.0f;
    }
    return 2.0f * (prec * rec) / (prec + rec);
}

//
// macro-averaged (avg across all classes)
//

f32 macro_precision(const u8 *true_labels, const u8 *predicted_labels, u64 length) {
    f32 total_precision = 0.0f;

    for (u8 class_id = 0; class_id < NUM_CLASSES; class_id++) {
        total_precision += precision(true_labels, predicted_labels, length, class_id);
    }

    return total_precision / (f32)NUM_CLASSES;
}

f32 macro_recall(const u8 *true_labels, const u8 *predicted_labels, u64 length) {
    f32 total_recall = 0.0f;

    for (u8 class_id = 0; class_id < NUM_CLASSES; class_id++) {
        total_recall += recall(true_labels, predicted_labels, length, class_id);
    }

    return total_recall / (f32)NUM_CLASSES;
}

f32 macro_f1_score(const u8 *true_labels, const u8 *predicted_labels, u64 length) {
    f32 total_f1 = 0.0f;

    for (u8 class_id = 0; class_id < NUM_CLASSES; class_id++) {
        total_f1 += f1_score(true_labels, predicted_labels, length, class_id);
    }

    return total_f1 / (f32)NUM_CLASSES;
}

//
// ascii tables
//

void print_metrics(const u8 *true_labels, const u8 *predicted_labels, u64 length) {
    f32 acc = accuracy(true_labels, predicted_labels, length);
    f32 prec = macro_precision(true_labels, predicted_labels, length);
    f32 rec = macro_recall(true_labels, predicted_labels, length);
    f32 f1 = macro_f1_score(true_labels, predicted_labels, length);

    printf("\n┌─────────────┬─────────┐\n");
    printf("│ metric      ┆ value   │\n");
    printf("│ ---         ┆ ---     │\n");
    printf("│ str         ┆ f32     │\n");
    printf("╞═════════════╪═════════╡\n");
    printf("│ accuracy    ┆ %.4f  │\n", acc);
    printf("│ precision   ┆ %.4f  │\n", prec);
    printf("│ recall      ┆ %.4f  │\n", rec);
    printf("│ f1_score    ┆ %.4f  │\n", f1);
    printf("└─────────────┴─────────┘\n");
}

void print_detailed_metrics(const u8 *true_labels, const u8 *predicted_labels, u64 length) {
    print_metrics(true_labels, predicted_labels, length);

    printf("\n┌────────────┬───────────┬────────┬──────────┐\n");
    printf("│ class      ┆ precision ┆ recall ┆ f1_score │\n");
    printf("│ ---        ┆ ---       ┆ ---    ┆ ---      │\n");
    printf("│ str        ┆ f32       ┆ f32    ┆ f32      │\n");
    printf("╞════════════╪═══════════╪════════╪══════════╡\n");

    for (u8 class_id = 0; class_id < NUM_CLASSES; class_id++) {
        f32 prec = precision(true_labels, predicted_labels, length, class_id);
        f32 rec = recall(true_labels, predicted_labels, length, class_id);
        f32 f1 = f1_score(true_labels, predicted_labels, length, class_id);

        printf("│ %-10s ┆ %-9.4f ┆ %-6.4f ┆ %-8.4f │\n", get_class_name(class_id), prec, rec, f1);
    }

    printf("└────────────┴───────────┴────────┴──────────┘\n");
}

void print_confusion_matrix(const u8 *true_labels, const u8 *predicted_labels, u64 length) {
    u32 confusion_matrix[NUM_CLASSES][NUM_CLASSES] = {0};

    for (u64 i = 0; i < length; i++) {
        confusion_matrix[true_labels[i]][predicted_labels[i]]++;
    }

    printf("\n┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐\n");
    printf("│ t\\p  ┆ airp ┆ auto ┆ bird ┆ cat  ┆ deer ┆ dog  ┆ frog ┆ hors ┆ ship ┆ truc │\n");
    printf("│ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---  ┆ ---  │\n");
    printf("│ str  ┆ u32  ┆ u32  ┆ u32  ┆ u32  ┆ u32  ┆ u32  ┆ u32  ┆ u32  ┆ u32  ┆ u32  │\n");
    printf("╞══════╪══════╪══════╪══════╪══════╪══════╪══════╪══════╪══════╪══════╪══════╡\n");

    const char *short_names[] = {"airp", "auto", "bird", "cat ", "deer", "dog ", "frog", "hors", "ship", "truc"};

    for (u8 i = 0; i < NUM_CLASSES; i++) {
        printf("│ %-4s ┆", short_names[i]);
        for (u8 j = 0; j < NUM_CLASSES; j++) {
            printf(" %-4u ┆", confusion_matrix[i][j]);
        }
        printf("\n");
    }

    printf("└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘\n");
}
