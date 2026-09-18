from src.schema import RegressionResult


def test_schema():

    result = RegressionResult(
        module="classic_ml",
        baseline_metric=0.90,
        candidate_metric=0.85,
        absolute_change=-0.05,
        relative_change=-0.055,
        p_value=0.01,
        significance_level=0.05,
        threshold=0.02,
        statistically_significant=True,
        practically_significant=True,
        status="FAIL"
    )

    assert result.module == "classic_ml"
    assert result.status == "FAIL"