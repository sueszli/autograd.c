#pragma once

#include "../datasets/cifar10.h"
#include "../utils/types.h"

f32 accuracy(const cifar10_class_t *true_labels, const cifar10_class_t *predicted_labels, u64 length);
f32 precision(const cifar10_class_t *true_labels, const cifar10_class_t *predicted_labels, u64 length, cifar10_class_t class_id);
f32 recall(const cifar10_class_t *true_labels, const cifar10_class_t *predicted_labels, u64 length, cifar10_class_t class_id);
f32 f1_score(const cifar10_class_t *true_labels, const cifar10_class_t *predicted_labels, u64 length, cifar10_class_t class_id);

f32 macro_precision(const cifar10_class_t *true_labels, const cifar10_class_t *predicted_labels, u64 length);
f32 macro_recall(const cifar10_class_t *true_labels, const cifar10_class_t *predicted_labels, u64 length);
f32 macro_f1_score(const cifar10_class_t *true_labels, const cifar10_class_t *predicted_labels, u64 length);

void print_metrics(const cifar10_class_t *true_labels, const cifar10_class_t *predicted_labels, u64 length);
void print_detailed_metrics(const cifar10_class_t *true_labels, const cifar10_class_t *predicted_labels, u64 length);
void print_confusion_matrix(const cifar10_class_t *true_labels, const cifar10_class_t *predicted_labels, u64 length);
