#include "../src/datasets/cifar10.h"
#include "../src/eval/metrics.h"
#include "../src/utils/types.h"
#include <unity.h>

static cifar10_class_t true_labels[] = {CAT, SHIP, SHIP, AIRPLANE};
static cifar10_class_t predicted_labels[] = {CAT, SHIP, CAT, AIRPLANE};
static u64 length = 4;

void setUp(void) {}

void tearDown(void) {}

void test_accuracy(void) {
    f32 acc = accuracy(true_labels, predicted_labels, length);
    TEST_ASSERT_EQUAL_FLOAT(0.75, acc);
}

void test_accuracy_empty(void) {
    f32 acc = accuracy(true_labels, predicted_labels, 0);
    TEST_ASSERT_EQUAL_FLOAT(0.0, acc);
}

void test_precision(void) {
    f32 prec_cat = precision(true_labels, predicted_labels, length, CAT);
    TEST_ASSERT_EQUAL_FLOAT(0.5, prec_cat);

    f32 prec_ship = precision(true_labels, predicted_labels, length, SHIP);
    TEST_ASSERT_EQUAL_FLOAT(1.0, prec_ship);

    f32 prec_airplane = precision(true_labels, predicted_labels, length, AIRPLANE);
    TEST_ASSERT_EQUAL_FLOAT(1.0, prec_airplane);

    f32 prec_dog = precision(true_labels, predicted_labels, length, DOG);
    TEST_ASSERT_EQUAL_FLOAT(0.0, prec_dog);
}

void test_recall(void) {
    f32 rec_cat = recall(true_labels, predicted_labels, length, CAT);
    TEST_ASSERT_EQUAL_FLOAT(1.0, rec_cat);

    f32 rec_ship = recall(true_labels, predicted_labels, length, SHIP);
    TEST_ASSERT_EQUAL_FLOAT(0.5, rec_ship);

    f32 rec_airplane = recall(true_labels, predicted_labels, length, AIRPLANE);
    TEST_ASSERT_EQUAL_FLOAT(1.0, rec_airplane);

    f32 rec_dog = recall(true_labels, predicted_labels, length, DOG);
    TEST_ASSERT_EQUAL_FLOAT(0.0, rec_dog);
}

void test_f1_score(void) {
    f32 f1_cat = f1_score(true_labels, predicted_labels, length, CAT);
    TEST_ASSERT_FLOAT_WITHIN(0.0001, 0.66666, f1_cat);

    f32 f1_ship = f1_score(true_labels, predicted_labels, length, SHIP);
    TEST_ASSERT_FLOAT_WITHIN(0.0001, 0.66666, f1_ship);

    f32 f1_airplane = f1_score(true_labels, predicted_labels, length, AIRPLANE);
    TEST_ASSERT_EQUAL_FLOAT(1.0, f1_airplane);

    f32 f1_dog = f1_score(true_labels, predicted_labels, length, DOG);
    TEST_ASSERT_EQUAL_FLOAT(0.0, f1_dog);
}

void test_macro_precision(void) {
    f32 macro_prec = macro_precision(true_labels, predicted_labels, length);
    TEST_ASSERT_EQUAL_FLOAT(0.25, macro_prec);
}

void test_macro_recall(void) {
    f32 macro_rec = macro_recall(true_labels, predicted_labels, length);
    TEST_ASSERT_EQUAL_FLOAT(0.25, macro_rec);
}

void test_macro_f1_score(void) {
    f32 macro_f1 = macro_f1_score(true_labels, predicted_labels, length);
    TEST_ASSERT_FLOAT_WITHIN(0.0001, 0.23333, macro_f1);
}

i32 main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_accuracy);
    RUN_TEST(test_accuracy_empty);
    RUN_TEST(test_precision);
    RUN_TEST(test_recall);
    RUN_TEST(test_f1_score);
    RUN_TEST(test_macro_precision);
    RUN_TEST(test_macro_recall);
    RUN_TEST(test_macro_f1_score);

    return UNITY_END();
}
