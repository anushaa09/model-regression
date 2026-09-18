from src.schema import RegressionResult


def detect_regression(
    module,
    baseline_metric,
    candidate_metric,
    p_value,
    significance_level=0.05,
    threshold=0.02,
    metric_name="accuracy",
    additional_metrics=None,
    slice_results=None
):
    """
    Detect model regression using:

    1. Statistical significance
    2. Overall metric thresholds
    3. Slice-level metric thresholds
    """

    absolute_change = (
        candidate_metric - baseline_metric
    )

    if baseline_metric != 0:
        relative_change = (
            absolute_change / baseline_metric
        )
    else:
        relative_change = 0.0

    # -----------------------------------------
    # Overall regression
    # -----------------------------------------

    statistically_significant = (
        p_value < significance_level
    )

    main_metric_regressed = (
        absolute_change <= -threshold
    )

    additional_regressions = {}

    if additional_metrics is not None:

        for metric in additional_metrics:

            name = metric["name"]

            baseline_value = metric["baseline"]
            candidate_value = metric["candidate"]
            metric_threshold = metric["threshold"]

            change = (
                candidate_value -
                baseline_value
            )

            additional_regressions[name] = {

                "baseline":
                    baseline_value,

                "candidate":
                    candidate_value,

                "change":
                    change,

                "threshold":
                    metric_threshold,

                "regressed":
                    change <= -metric_threshold
            }

    overall_practical_regression = (
        main_metric_regressed
        or any(
            metric["regressed"]
            for metric
            in additional_regressions.values()
        )
    )

    # -----------------------------------------
    # Slice regression
    # -----------------------------------------

    slice_regressions = {}

    if slice_results is not None:

        for group, metrics in slice_results.items():

            accuracy_regressed = (
                metrics["accuracy_drop"]
                >= threshold
            )

            f1_regressed = (
                metrics["f1_drop"]
                >= 0.02
            )

            slice_regressions[group] = {

                "accuracy_regressed":
                    accuracy_regressed,

                "f1_regressed":
                    f1_regressed,

                "regressed":
                    (
                        accuracy_regressed
                        or f1_regressed
                    )
            }

    slice_regression_detected = any(
        item["regressed"]
        for item in slice_regressions.values()
    )

    # -----------------------------------------
    # Final decision
    # -----------------------------------------

    regression = (
        (
            statistically_significant
            and overall_practical_regression
        )
        or slice_regression_detected
    )

    status = (
        "FAIL"
        if regression
        else "PASS"
    )

    # -----------------------------------------
    # Details
    # -----------------------------------------

    details = {

        "main_metric":
            metric_name,

        "additional_metrics":
            additional_regressions,

        "slice_regressions":
            slice_regressions
    }

    return RegressionResult(

        module=module,

        baseline_metric=
            float(baseline_metric),

        candidate_metric=
            float(candidate_metric),

        absolute_change=
            float(absolute_change),

        relative_change=
            float(relative_change),

        p_value=
            float(p_value),

        significance_level=
            float(significance_level),

        threshold=
            float(threshold),

        statistically_significant=
            statistically_significant,

        practically_significant=(
            overall_practical_regression
            or slice_regression_detected
        ),

        status=status,

        details=details
    )