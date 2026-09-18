# 🤖 Model Regression Detector

An MLOps system for automatically detecting performance regressions between a trusted baseline machine learning model and a candidate model version.

The system evaluates both models on the same fixed golden test set and combines:

- Accuracy
- F1-score
- McNemar's statistical significance test
- Practical performance thresholds
- Age-based slice analysis
- Automated PASS/FAIL regression decisions
- Evaluation history
- FastAPI API
- Streamlit dashboard
- GitHub Actions CI regression gating

---

## 🎯 Problem

When a machine learning model is updated, a new version may appear to work correctly while actually performing worse than the existing production model.

Traditional unit tests cannot reliably detect this type of ML regression.

For example:

```text
Baseline Accuracy:   86.44%
Candidate Accuracy:  84.85%

Baseline F1:         67.04%
Candidate F1:        58.37%
