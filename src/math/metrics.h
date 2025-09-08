#pragma once

#include "../utils/types.h"

f32 accuracy(const u8 *true_labels, const u8 *predicted_labels, u64 length);
f32 precision(const u8 *true_labels, const u8 *predicted_labels, u64 length, u8 class_id);
f32 recall(const u8 *true_labels, const u8 *predicted_labels, u64 length, u8 class_id);
f32 f1_score(const u8 *true_labels, const u8 *predicted_labels, u64 length, u8 class_id);

f32 macro_precision(const u8 *true_labels, const u8 *predicted_labels, u64 length);
f32 macro_recall(const u8 *true_labels, const u8 *predicted_labels, u64 length);
f32 macro_f1_score(const u8 *true_labels, const u8 *predicted_labels, u64 length);

void print_metrics(const u8 *true_labels, const u8 *predicted_labels, u64 length);
void print_detailed_metrics(const u8 *true_labels, const u8 *predicted_labels, u64 length);
void print_confusion_matrix(const u8 *true_labels, const u8 *predicted_labels, u64 length);
