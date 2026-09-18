from src.regression_engine import detect_regression


def test_regression_detected():

    result = detect_regression(
        module="classic_ml",
        baseline_metric=0.90,
        candidate_metric=0.85,
        p_value=0.01,
        significance_level=0.05,
        threshold=0.02
    )

    assert result.status == "FAIL"


def test_small_drop_passes():

    result = detect_regression(
        module="classic_ml",
        baseline_metric=0.90,
        candidate_metric=0.895,
        p_value=0.01,
        significance_level=0.05,
        threshold=0.02
    )

    assert result.status == "PASS"


def test_not_significant_passes():

    result = detect_regression(
        module="classic_ml",
        baseline_metric=0.90,
        candidate_metric=0.85,
        p_value=0.20,
        significance_level=0.05,
        threshold=0.02
    )

    assert result.status == "PASS"