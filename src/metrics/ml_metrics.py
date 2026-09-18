import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import chi2


def calculate_accuracy(y_true, predictions):
    return accuracy_score(y_true, predictions)


def calculate_f1(y_true, predictions):
    return f1_score(y_true, predictions)


def mcnemar_test(
    y_true,
    baseline_pred,
    candidate_pred
):
    """
    McNemar's test using
    continuity-corrected chi-square.
    """

    baseline_correct = (
        np.array(baseline_pred)
        == np.array(y_true)
    )

    candidate_correct = (
        np.array(candidate_pred)
        == np.array(y_true)
    )

    b = np.sum(
        baseline_correct & ~candidate_correct
    )

    c = np.sum(
        ~baseline_correct & candidate_correct
    )

    if b + c == 0:
        return 1.0

    statistic = (
        (abs(b - c) - 1) ** 2
        / (b + c)
    )

    p_value = chi2.sf(
        statistic,
        df=1
    )

    return float(p_value)


def subgroup_accuracy(
    y_true,
    predictions,
    groups
):
    results = {}

    y_true = np.array(y_true)
    predictions = np.array(predictions)
    groups = np.array(groups)

    for group in np.unique(groups):

        mask = groups == group

        results[str(group)] = {
            "accuracy": float(
                accuracy_score(
                    y_true[mask],
                    predictions[mask]
                )
            ),
            "samples": int(mask.sum())
        }

    return results


def compare_slices(
    y_true,
    baseline_pred,
    candidate_pred,
    groups
):
    results = {}

    y_true = np.array(y_true)
    baseline_pred = np.array(baseline_pred)
    candidate_pred = np.array(candidate_pred)
    groups = np.array(groups)

    for group in np.unique(groups):

        mask = groups == group

        # Ignore very small groups
        if mask.sum() < 50:
            continue

        baseline_accuracy = accuracy_score(
            y_true[mask],
            baseline_pred[mask]
        )

        candidate_accuracy = accuracy_score(
            y_true[mask],
            candidate_pred[mask]
        )

        baseline_f1 = f1_score(
            y_true[mask],
            baseline_pred[mask]
        )

        candidate_f1 = f1_score(
            y_true[mask],
            candidate_pred[mask]
        )

        results[str(group)] = {

            "samples": int(mask.sum()),

            "baseline_accuracy":
                float(baseline_accuracy),

            "candidate_accuracy":
                float(candidate_accuracy),

            "accuracy_drop":
                float(
                    baseline_accuracy
                    - candidate_accuracy
                ),

            "baseline_f1":
                float(baseline_f1),

            "candidate_f1":
                float(candidate_f1),

            "f1_drop":
                float(
                    baseline_f1
                    - candidate_f1
                )
        }

    return results