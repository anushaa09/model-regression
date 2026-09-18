
# 🤖 Model Regression Detector

An MLOps system that automatically detects performance regressions between a trusted baseline machine learning model and a candidate model version.

The system evaluates both models on the same fixed golden test set and combines statistical testing, performance thresholds, and slice-level analysis to produce an automated **PASS/FAIL** decision.

---

## 🚀 Live Demo

### 🌐 Streamlit Dashboard

https://model-regression-7rfhqehrqvtqobm5v3hqve.streamlit.app/

### ⚡ FastAPI Backend

https://model-regression-r55s.onrender.com/

### 📚 API Documentation

https://model-regression-r55s.onrender.com/docs

---

## 🎯 Problem

When a machine learning model is updated, the new model may appear to work correctly while actually performing worse than the existing trusted model.

Traditional software tests are not sufficient to detect many ML-specific regressions because the code can work correctly even when model performance decreases.

For example:

```text
Baseline Accuracy:   86.44%
Candidate Accuracy:  84.85%

Baseline F1:         67.04%
Candidate F1:        58.37%
````

The Model Regression Detector automatically evaluates the new model against the baseline and determines whether the candidate model should pass or fail the regression check.

---

# 🏗️ System Architecture

```text
                         GitHub Repository
                                │
                                ▼
                       GitHub Actions CI
                                │
                ┌───────────────┴────────────────┐
                │                                │
                ▼                                ▼
          Run Unit Tests                   Train Models
                                                  │
                                      ┌───────────┴───────────┐
                                      ▼                       ▼
                              Baseline Model          Candidate Model
                                      │                       │
                                      └───────────┬───────────┘
                                                  │
                                                  ▼
                                        Golden Test Dataset
                                                  │
                                                  ▼
                                      Regression Evaluation
                                                  │
                     ┌────────────────────────────┼────────────────────────┐
                     │                            │                        │
                     ▼                            ▼                        ▼
                 Accuracy                     F1-score              McNemar Test
                     │                            │                        │
                     └────────────────────────────┼────────────────────────┘
                                                  │
                                                  ▼
                                           Slice Analysis
                                                  │
                                                  ▼
                                           PASS / FAIL
                                                  │
                              ┌───────────────────┴───────────────────┐
                              │                                       │
                              ▼                                       ▼
                         FastAPI Backend                       Streamlit Dashboard
                              │                                       │
                              └───────────────────┬───────────────────┘
                                                  ▼
                                           Deployed System
```

---

# ⚙️ How It Works

The system follows a model regression testing pipeline.

### 1. Train the baseline model

The baseline model represents the trusted model version.

```text
models/baseline_model.pkl
```

### 2. Train the candidate model

The candidate model represents the new model version being evaluated.

```text
models/candidate_model.pkl
```

### 3. Use a fixed golden test set

Both models are evaluated on the same test dataset:

```text
data/golden_sets/adult_income_test.csv
```

Using the same test set ensures that the comparison is performed under identical evaluation conditions.

### 4. Calculate performance metrics

The system calculates:

* Accuracy
* F1-score

for both model versions.

### 5. Run McNemar's statistical test

McNemar's test compares the prediction correctness of the two models.

This provides statistical evidence about whether the models behave differently on the evaluation set.

### 6. Apply practical thresholds

The system checks whether the performance difference exceeds predefined thresholds.

Default configuration:

```env
SIGNIFICANCE_LEVEL=0.05
MIN_ACCURACY_DROP=0.02
MIN_F1_DROP=0.02
```

### 7. Perform slice analysis

The system evaluates model performance across age groups:

```text
<30
30-50
>50
```

This helps identify regressions that may not be obvious from overall metrics.

### 8. Produce a final decision

The system produces either:

```text
PASS
```

or:

```text
FAIL
```

A failed regression causes the CI pipeline to fail.

---

# 📊 Example Evaluation

The current candidate model intentionally represents a weaker model version.

## Overall Metrics

| Metric   | Baseline | Candidate |   Change |
| -------- | -------: | --------: | -------: |
| Accuracy |   86.44% |    84.85% | -1.60 pp |
| F1-score |   67.04% |    58.37% | -8.67 pp |

## Statistical Test

```text
McNemar p-value: 7.65 × 10⁻⁹
Significance level: 0.05
```

The prediction difference is statistically significant under the configured significance level.

## Age Slice Analysis

| Age Group | Samples | Baseline Accuracy | Candidate Accuracy | Accuracy Drop | Baseline F1 | Candidate F1 |  F1 Drop |
| --------- | ------: | ----------------: | -----------------: | ------------: | ----------: | -----------: | -------: |
| <30       |   1,938 |            96.80% |             95.87% |       0.93 pp |      59.74% |       33.33% | 26.41 pp |
| 30-50     |   3,259 |            82.42% |             80.64% |       1.78 pp |      67.65% |       59.94% |  7.71 pp |
| >50       |   1,316 |            81.16% |             79.03% |       2.13 pp |      67.11% |       59.17% |  7.94 pp |

The resulting decision is:

```text
🚨 FAIL — Regression Detected
```

---

# 🔬 Regression Detection Methodology

The detector combines multiple signals rather than relying on a single metric.

## Statistical Significance

McNemar's test is applied to the paired predictions produced by the baseline and candidate models.

This evaluates whether their prediction correctness differs significantly.

## Practical Significance

Performance changes are compared against configurable thresholds.

For example:

```text
Accuracy threshold = 2 percentage points
F1 threshold       = 2 percentage points
```

A metric is considered practically regressed when its drop reaches or exceeds the configured threshold.

## Slice-Level Analysis

The system also evaluates performance across age groups.

This helps expose localized degradation that can be hidden by aggregate metrics.

---

# 🚦 CI/CD Regression Gate

The project uses **GitHub Actions** to automatically run the regression testing pipeline.

The CI workflow performs:

```text
1. Checkout repository
2. Set up Python
3. Install dependencies
4. Run unit tests
5. Train baseline model
6. Train candidate model
7. Run model regression evaluation
```

The evaluation script contains the regression gate:

```python
if result.status == "FAIL":
    raise SystemExit(1)
```

Therefore:

```text
Regression detected
        ↓
Exit code 1
        ↓
GitHub Actions job fails
```

while:

```text
No regression
        ↓
Exit code 0
        ↓
GitHub Actions job passes
```

This allows model quality checks to become part of the CI/CD workflow.

---

# 🧪 Testing

The project uses **pytest** for automated testing.

Tests cover:

* Regression result schema
* Regression detection logic
* Accuracy calculation
* F1-score calculation
* McNemar's test
* Slice comparison

Run the test suite:

```bash
python -m pytest
```

Current result:

```text
8 passed
```

---

# 🌐 API

The FastAPI backend exposes the following endpoints.

## `GET /`

Checks that the API is running.

Example response:

```json
{
  "message": "Model Regression Detector API"
}
```

## `POST /run-eval`

Runs a complete model regression evaluation.

The endpoint:

1. Loads the baseline model
2. Loads the candidate model
3. Loads the golden test set
4. Generates predictions
5. Calculates Accuracy and F1
6. Runs McNemar's test
7. Performs age-based slice analysis
8. Determines PASS/FAIL
9. Stores the evaluation run
10. Returns the evaluation result

## `GET /history`

Returns previous evaluation runs stored in SQLite.

---

# 📊 Streamlit Dashboard

The Streamlit dashboard provides a visual interface for the regression testing system.

The dashboard displays:

* Regression status
* Baseline accuracy
* Candidate accuracy
* Accuracy change
* Baseline F1
* Candidate F1
* F1 change
* McNemar p-value
* Age slice analysis
* Evaluation history
* Full evaluation details

### Dashboard Flow

```text
Click "Run Model Evaluation"
            ↓
       FastAPI API
            ↓
     Regression Engine
            ↓
       Model Results
            ↓
     Streamlit Dashboard
```

---

# 💾 Evaluation History

Each evaluation run is stored using SQLite.

Stored information includes:

* Run ID
* Module
* Baseline metric
* Candidate metric
* McNemar p-value
* PASS/FAIL status
* Timestamp

This allows previous model evaluations to be reviewed through the dashboard.

---

# 📁 Project Structure

```text
model-regression/
│
├── .github/
│   └── workflows/
│       └── regression-check.yml
│
├── data/
│   ├── raw/
│   │   └── adult_income.csv
│   │
│   ├── golden_sets/
│   │   └── adult_income_test.csv
│   │
│   └── run_history.db
│
├── models/
│   ├── baseline_model.pkl
│   └── candidate_model.pkl
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   │
│   ├── dashboard/
│   │   └── app.py
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── ml_metrics.py
│   │
│   ├── runners/
│   │   ├── __init__.py
│   │   └── ml_runner.py
│   │
│   ├── __init__.py
│   ├── config.py
│   ├── db.py
│   ├── regression_engine.py
│   ├── report.py
│   └── schema.py
│
├── scripts/
│   ├── download_data.py
│   ├── train_baseline.py
│   ├── train_candidate.py
│   └── run_evaluation.py
│
├── tests/
│   ├── __init__.py
│   ├── test_ml_metrics.py
│   ├── test_regression_engine.py
│   └── test_schema.py
│
├── .env.example
├── .gitignore
├── render.yaml
├── requirements.txt
└── README.md
```

---

# 🛠️ Tech Stack

## Machine Learning

* Python
* Scikit-learn
* Pandas
* NumPy
* SciPy
* Joblib

## MLOps

* Model regression testing
* Golden test datasets
* Statistical model comparison
* Slice-level evaluation
* CI/CD quality gates
* Automated evaluation

## Backend

* FastAPI
* Uvicorn
* SQLAlchemy
* SQLite

## Dashboard

* Streamlit

## Testing

* Pytest

## Deployment

* Render
* Streamlit Community Cloud

## Version Control / CI

* Git
* GitHub
* GitHub Actions

---

# 🚀 Running Locally

## 1. Clone the repository

```bash
git clone https://github.com/anushaa09/model-regression.git
cd model-regression
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Train the baseline model

```bash
python -m scripts.train_baseline
```

## 5. Train the candidate model

```bash
python -m scripts.train_candidate
```

## 6. Run regression evaluation

```bash
python -m scripts.run_evaluation
```

## 7. Run tests

```bash
python -m pytest
```

---

# ⚡ Running the FastAPI Backend

Start the API:

```bash
python -m uvicorn src.api.main:app --reload
```

Open:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

# 📊 Running the Streamlit Dashboard

Start Streamlit:

```bash
streamlit run src/dashboard/app.py
```

The dashboard uses:

```env
API_URL=http://localhost:8000
```

by default.

For cloud deployment, `API_URL` can be configured to point to the deployed FastAPI backend.

---

# ☁️ Deployment

The FastAPI backend is deployed using Render.

The Streamlit dashboard is deployed using Streamlit Community Cloud.

### Production architecture

```text
                    User
                     │
                     ▼
            Streamlit Dashboard
                     │
                     │ HTTP API
                     ▼
              Render FastAPI
                     │
                     ▼
         Model Regression Engine
                     │
             ┌───────┴───────┐
             ▼               ▼
        Baseline Model   Candidate Model
             │               │
             └───────┬───────┘
                     ▼
              Golden Dataset
                     │
                     ▼
              Regression Result
```

---

# 🔐 Configuration

Regression thresholds can be configured using environment variables.

Example:

```env
SIGNIFICANCE_LEVEL=0.05
MIN_ACCURACY_DROP=0.02
MIN_F1_DROP=0.02
```

### Configuration

| Variable             | Default | Description                                |
| -------------------- | ------: | ------------------------------------------ |
| `SIGNIFICANCE_LEVEL` |  `0.05` | Statistical significance level             |
| `MIN_ACCURACY_DROP`  |  `0.02` | Minimum accuracy drop considered practical |
| `MIN_F1_DROP`        |  `0.02` | Minimum F1 drop considered practical       |

---

# 🧠 Dataset

The project uses the **UCI Adult Income dataset**.

The dataset contains demographic and employment-related features and is used here as a classification benchmark for demonstrating model regression testing.

The target variable represents whether an individual's income exceeds the dataset's defined income threshold.

A fixed test split is saved as the project's golden test set so that baseline and candidate models are evaluated under identical conditions.

---

# 💡 Why This Project?

Machine learning systems require more than checking whether code executes successfully.

A new model version can introduce:

* Lower predictive performance
* Metric regressions
* Changes in prediction behavior
* Slice-specific degradation

This project treats model evaluation as a form of **regression testing** and integrates it directly into a CI/CD workflow.

Instead of simply asking:

```text
"Does the new model run?"
```

the system asks:

```text
"Did the new model regress compared with the trusted model?"
```

---

# 🎓 Key MLOps Concepts Demonstrated

This project demonstrates practical concepts including:

* Model version comparison
* Golden datasets
* Automated model evaluation
* Statistical testing
* Metric thresholds
* Slice-based evaluation
* Regression detection
* CI/CD automation
* Automated quality gates
* API-based model evaluation
* Evaluation history
* Cloud deployment
* Automated testing

---

# 🔮 Future Improvements

Potential improvements include:

* Additional statistical tests
* More configurable data slices
* Model version tracking
* Experiment tracking
* Model registry integration
* Persistent cloud database
* Automated model comparison reports
* Authentication for the API
* Docker containerization
* Production monitoring
* Data drift detection
* Model drift detection
* Integration with cloud ML platforms

---

# 👩‍💻 Author

## Anusha

Computer Science & Engineering
VIT Vellore

---

## ⭐ Project Highlights

```text
✓ ML Regression Testing
✓ Baseline vs Candidate Comparison
✓ Accuracy & F1 Evaluation
✓ McNemar Statistical Test
✓ Age-Based Slice Analysis
✓ Automated PASS/FAIL Decision
✓ GitHub Actions CI/CD Gate
✓ FastAPI Backend
✓ Streamlit Dashboard
✓ SQLite Evaluation History
✓ Automated Unit Tests
✓ Cloud Deployment
```

---

## 📌 Repository

GitHub:

[https://github.com/anushaa09/model-regression](https://github.com/anushaa09/model-regression)

````

### After pasting it

Run:

```powershell
git add README.md
git commit -m "Add comprehensive project documentation"
git push
````
