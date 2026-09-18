import joblib
import pandas as pd

from src.config import (
    BASELINE_MODEL_PATH,
    CANDIDATE_MODEL_PATH,
    TEST_DATA_PATH,
    SIGNIFICANCE_LEVEL,
    MIN_ACCURACY_DROP,
    MIN_F1_DROP
)

from src.metrics.ml_metrics import (
    calculate_accuracy,
    calculate_f1,
    mcnemar_test,
    compare_slices
)

from src.regression_engine import detect_regression


def run_ml_evaluation():

    # 1. Load models

    baseline_model = joblib.load(
        BASELINE_MODEL_PATH
    )

    candidate_model = joblib.load(
        CANDIDATE_MODEL_PATH
    )

    # 2. Load golden test set

    df = pd.read_csv(
        TEST_DATA_PATH
    )

    X = df.drop(
        columns=["target"]
    )

    y = df["target"]

    # 3. Predictions

    baseline_pred = baseline_model.predict(X)

    candidate_pred = candidate_model.predict(X)

    # 4. Accuracy

    baseline_accuracy = calculate_accuracy(
        y,
        baseline_pred
    )

    candidate_accuracy = calculate_accuracy(
        y,
        candidate_pred
    )

    # 5. F1

    baseline_f1 = calculate_f1(
        y,
        baseline_pred
    )

    candidate_f1 = calculate_f1(
        y,
        candidate_pred
    )

    # 6. McNemar test

    p_value = mcnemar_test(
        y,
        baseline_pred,
        candidate_pred
    )

    # 7. Age slices

    age_groups = []

    for value in df["age"]:

        if value < 30:
            age_groups.append("<30")

        elif value <= 50:
            age_groups.append("30-50")

        else:
            age_groups.append(">50")

    slice_results = compare_slices(
        y,
        baseline_pred,
        candidate_pred,
        age_groups
    )

    # 8. Detect regression

    result = detect_regression(

        module="classic_ml",

        baseline_metric=
            baseline_accuracy,

        candidate_metric=
            candidate_accuracy,

        p_value=p_value,

        significance_level=
            SIGNIFICANCE_LEVEL,

        threshold=
            MIN_ACCURACY_DROP,

        metric_name="accuracy",

        additional_metrics=[

            {
                "name": "f1",

                "baseline":
                    baseline_f1,

                "candidate":
                    candidate_f1,

                "threshold":
                    MIN_F1_DROP
            }
        ],

        slice_results=
            slice_results
    )

    # 9. Add details

    result.details.update({

        "baseline_accuracy":
            baseline_accuracy,

        "candidate_accuracy":
            candidate_accuracy,

        "accuracy_drop":
            baseline_accuracy -
            candidate_accuracy,

        "baseline_f1":
            baseline_f1,

        "candidate_f1":
            candidate_f1,

        "f1_drop":
            baseline_f1 -
            candidate_f1,

        "test_samples":
            len(df),

        "slices":
            slice_results
    })

    return result