from src.metrics.ml_metrics import (
    calculate_accuracy,
    calculate_f1,
    mcnemar_test,
    compare_slices
)


def test_accuracy():

    y_true = [0, 1, 1, 0]
    predictions = [0, 1, 0, 0]

    result = calculate_accuracy(
        y_true,
        predictions
    )

    assert result == 0.75


def test_f1():

    y_true = [0, 1, 1, 0]
    predictions = [0, 1, 0, 0]

    result = calculate_f1(
        y_true,
        predictions
    )

    assert result > 0


def test_mcnemar():

    y_true = [0, 1, 1, 0]

    baseline = [0, 1, 1, 0]

    candidate = [1, 1, 0, 0]

    result = mcnemar_test(
        y_true,
        baseline,
        candidate
    )

    assert 0 <= result <= 1


def test_compare_slices():

    
    y_true = (
        [0] * 25 +
        [1] * 25
    )

    baseline = (
        [0] * 25 +
        [1] * 25
    )

    candidate = (
        [0] * 25 +
        [0] * 25
    )

    groups = ["A"] * 50

    result = compare_slices(
        y_true,
        baseline,
        candidate,
        groups
    )

    assert "A" in result
    assert result["A"]["samples"] == 50

    assert "A" in result
    assert result["A"]["samples"] == 50