import json

from src.schema import RegressionResult


def result_to_markdown(result: RegressionResult):

    status_icon = (
        "❌"
        if result.status == "FAIL"
        else "✅"
    )

    report = f"""
# Model Regression Report

## Overall Status

{status_icon} **{result.status}**

## Overall Metrics

| Metric | Baseline | Candidate | Drop |
|---|---:|---:|---:|
| Accuracy | {result.details["baseline_accuracy"]:.2%} | {result.details["candidate_accuracy"]:.2%} | {result.details["accuracy_drop"]:.2%} |
| F1 | {result.details["baseline_f1"]:.2%} | {result.details["candidate_f1"]:.2%} | {result.details["f1_drop"]:.2%} |

## Statistical Test

**McNemar p-value:** {result.p_value:.6f}

**Significance level:** {result.significance_level}

Statistically significant:

**{result.statistically_significant}**

## Slice Analysis

| Age Group | Samples | Accuracy Drop | F1 Drop | Status |
|---|---:|---:|---:|---|
"""

    slices = result.details.get("slices", {})

    for group, metrics in slices.items():

        accuracy_drop = metrics["accuracy_drop"]
        f1_drop = metrics["f1_drop"]

        if (
            accuracy_drop >= result.threshold
            or f1_drop >= 0.02
        ):
            status = "🚨 REGRESSION"
        else:
            status = "✅ OK"

        report += (
            f"| {group} "
            f"| {metrics['samples']} "
            f"| {accuracy_drop:.2%} "
            f"| {f1_drop:.2%} "
            f"| {status} |\n"
        )

    report += f"""

## Final Decision

Practically significant:

**{result.practically_significant}**

Regression detected:

**{result.status == "FAIL"}**

"""

    return report


def result_to_json(result: RegressionResult):

    return result.model_dump_json(
        indent=2
    )