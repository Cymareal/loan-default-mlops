# Loan Default MLOps Pipeline

A production-ready MLOps pipeline built on top of a loan default prediction model. This project demonstrates the full ML engineering lifecycle — from versioned data and tracked experiments to automated CI/CD and drift monitoring.

> **Portfolio Project 4 of 4** — AI/ML Engineering Portfolio by [Cymareal](https://github.com/Cymareal)

---

## What This Project Does

Takes the loan default prediction model from Project 1 and wraps it in a complete MLOps system:

- **Versioned data** — raw and processed datasets tracked with DVC, not Git
- **Experiment tracking** — every training run logs parameters, metrics, and model artifacts to MLflow
- **REST API** — Flask endpoint serves predictions with engineered features at inference time
- **Automated tests** — pytest suite with mocked model for CI compatibility
- **CI/CD pipeline** — GitHub Actions runs tests on every push and deploys on success
- **Drift monitoring** — Evidently AI compares production data distributions against training data and alerts on drift

---

## Project Structure

```
loan-default-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD pipeline
├── data/
│   ├── raw/
│   │   └── Loan_default.csv.dvc    # DVC pointer to raw dataset
│   └── processed/
│       └── features.csv.dvc        # DVC pointer to processed features
├── models/
│   └── voting_clf.pkl          # Trained ensemble model + threshold (gitignored)
├── notebooks/
│   └── loan_default_notebook.ipynb # Original Project 1 exploration
├── reports/
│   └── drift_report.html       # Evidently drift report (auto-generated)
├── src/
│   ├── preprocess.py           # Data cleaning and feature engineering
│   ├── train.py                # Model training with MLflow tracking
│   ├── predict.py              # Flask prediction API
│   └── monitor.py              # Drift detection and monitoring
├── tests/
│   └── test_predict.py         # Automated API tests
├── Dockerfile                  # Container definition
├── requirements.txt            # Pinned dependencies
└── README.md
```

---

## The Four Phases

### Phase 1 — Data Versioning (DVC)

Raw and processed data are tracked with DVC instead of Git. Git only stores small pointer files (`.dvc`) containing a hash fingerprint of the data. The actual CSVs live in the DVC cache.

```bash
# Initialize DVC
python -m dvc init

# Version the raw dataset
python -m dvc add data/raw/Loan_default.csv

# Run preprocessing and version the output
python src/preprocess.py
python -m dvc add data/processed/features.csv

# Commit the pointers (not the data)
git add .
git commit -m "feat: version raw and processed data"
```

**Why this matters:** Any experiment run is fully reproducible. You can `git checkout` any past commit and `dvc checkout` to restore the exact dataset that existed at that point in time.

---

### Phase 2 — Experiment Tracking (MLflow)

Every training run logs hyperparameters, evaluation metrics, and the serialized model to MLflow. Results are viewable in a browser UI.

```bash
# Run preprocessing
python src/preprocess.py

# Train both models and log everything to MLflow
python src/train.py

# Launch the MLflow UI
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://127.0.0.1:5000
```

**What gets logged per run:**
- Random Forest best params (n_estimators, max_depth, min_samples_split, min_samples_leaf)
- XGBoost best params (learning_rate, subsample, colsample_bytree, scale_pos_weight)
- Metrics: ROC-AUC, Recall, Precision, F1, optimal threshold
- Serialized VotingClassifier artifact

**Results:**

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.691 |
| Recall | 0.795 |
| Precision | 0.167 |
| F1 | 0.276 |
| Threshold | 0.50 |

The model is optimized for recall — in credit risk, missing a defaulter is far more costly than a false alarm.

---

### Phase 3 — CI/CD (GitHub Actions + Docker)

A push to `master` automatically triggers the pipeline:

1. Spins up a fresh Ubuntu container on GitHub's cloud
2. Installs Python 3.10 and dependencies
3. Runs all pytest tests
4. If tests pass, triggers deployment to Render

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/ -v

  deploy:
    needs: test   # only runs if tests pass
    ...
```

**Running the API locally:**

```bash
python src/predict.py
```

```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "Income": 50000,
    "CreditScore": 650,
    "LoanAmount": 15000,
    "LoanTerm": 36,
    "InterestRate": 12.5,
    "DTIRatio": 0.4,
    "NumCreditLines": 3,
    "MonthsEmployed": 24,
    "Education": 1,
    "EmploymentType": 1,
    "MaritalStatus": 1,
    "HasMortgage": 0,
    "HasDependents": 1,
    "LoanPurpose": 1,
    "HasCoSigner": 0
  }'
```

**Example response:**

```json
{
  "default_probability": 0.8051,
  "prediction": 1,
  "prediction_label": "Default",
  "threshold_used": 0.5
}
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Predict loan default probability |

**Running with Docker:**

```bash
docker build -t loan-default-mlops .
docker run -p 5001:5001 loan-default-mlops
```

---

### Phase 4 — Drift Monitoring (Evidently AI)

Compares incoming production data distributions against training data. Generates an HTML report and logs results to MLflow. Triggers a retraining alert if drift exceeds the threshold.

```bash
python src/monitor.py
```

**Output:**

```
=== Simulating DRIFTED production data ===
Report saved to reports/drift_report.html

--- Drift Monitoring Results ---
  Drift detected:  True
  Drifted columns: 5
  Drift share:     31.25%

ALERT: Data drift detected — model retraining recommended.

=== Simulating CLEAN production data ===
Report saved to reports/drift_report.html

--- Drift Monitoring Results ---
  Drift detected:  False
  Drifted columns: 0
  Drift share:     0.00%

No significant drift detected — model is healthy.
```

The HTML report shows per-column drift scores, distribution comparisons, and a dataset-level summary. All results are also logged to MLflow under the `loan-default-monitoring` experiment.

---

## Setup

### Prerequisites

- Python 3.10
- Git
- Docker (optional, for containerized deployment)

### Installation

```bash
git clone https://github.com/Cymareal/loan-default-mlops.git
cd loan-default-mlops

python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
# or
source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

### Data

The dataset is tracked with DVC. To pull it (requires DVC remote access):

```bash
dvc pull
```

Or download `Loan_default.csv` from [Kaggle](https://www.kaggle.com) and place it in `data/raw/`.

### Run the full pipeline

```bash
# 1. Preprocess
python src/preprocess.py

# 2. Train
python src/train.py

# 3. Serve
python src/predict.py

# 4. Monitor
python src/monitor.py
```

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Model Architecture

The ensemble combines two models trained with SMOTE-balanced data and tuned with RandomizedSearchCV optimizing for recall:

```
Input Features (16)
       │
       ▼
Feature Engineering
  ├── LoanToIncome = LoanAmount / (Income + 1)
  ├── CreditLinesPerYear = NumCreditLines / (MonthsEmployed/12 + 1)
  └── RiskInteraction = InterestRate × DTIRatio
       │
       ▼
SMOTE Oversampling (training only)
       │
       ├──► Random Forest (RandomizedSearchCV, recall-optimized)
       │
       └──► XGBoost (RandomizedSearchCV, recall-optimized)
              │
              ▼
       Soft Voting Ensemble
              │
              ▼
       Threshold Optimization
       (maximize precision at recall ≥ 0.70)
              │
              ▼
       Final Prediction
```

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Data versioning | DVC |
| Experiment tracking | MLflow |
| ML models | scikit-learn, XGBoost |
| Class imbalance | imbalanced-learn (SMOTE) |
| API | Flask + Gunicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Drift monitoring | Evidently AI |
| Testing | pytest |

---

## Portfolio Context

This is Project 4 of a 4-project AI/ML Engineering portfolio:

| # | Project | Stack | Status |
|---|---------|-------|--------|
| 1 | [Loan Default Prediction](https://github.com/Cymareal/loan-default-prediction) | Random Forest, XGBoost, SMOTE | ✅ Live |
| 2 | [Sentiment Analysis API](https://github.com/Cymareal/sentiment-analysis-api) | TF-IDF, Logistic Regression, Flask | ✅ Live |
| 3 | [Weather Image Classifier](https://github.com/Cymareal/weather-classifier) | MobileNetV2, TensorFlow, Grad-CAM | ✅ Live |
| 4 | Loan Default MLOps Pipeline | DVC, MLflow, Docker, GitHub Actions, Evidently | ✅ This repo |
