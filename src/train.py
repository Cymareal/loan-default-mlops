import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                             precision_score, classification_report)
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE
import os
import pickle

DATA_PATH = "data/processed/features.csv"
MODEL_DIR = "models"
MLFLOW_EXPERIMENT = "loan-default-prediction"

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Default"])
    y = df["Default"]
    return X, y

def find_best_threshold(y_test, y_prob):
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        precision = precision_score(y_test, y_pred_t, zero_division=0)
        recall    = recall_score(y_test, y_pred_t)
        f1        = f1_score(y_test, y_pred_t, zero_division=0)
        results.append([t, precision, recall, f1])
    results_df = pd.DataFrame(results, columns=["Threshold", "Precision", "Recall", "F1"])
    filtered = results_df[results_df["Recall"] >= 0.70]
    best_row = filtered.loc[filtered["Precision"].idxmax()]
    return best_row["Threshold"], results_df

def train():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    scale = (y_train == 0).sum() / (y_train == 1).sum()

    # Sample for RandomizedSearchCV speed
    sample_idx = np.random.choice(len(X_train_sm), size=min(5000, len(X_train_sm)), replace=False)
    X_sample = X_train_sm.iloc[sample_idx]
    y_sample = y_train_sm.iloc[sample_idx]

    # Random Forest search
    param_dist_rf = {
        "n_estimators":      randint(100, 300),
        "max_depth":         randint(10, 30),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf":  randint(1, 10)
    }
    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    search_rf = RandomizedSearchCV(rf, param_dist_rf, n_iter=10, cv=3,
                                   scoring="recall", random_state=42, n_jobs=-1, verbose=1)
    search_rf.fit(X_sample, y_sample)
    best_rf = search_rf.best_estimator_
    print("Best RF params:", search_rf.best_params_)

    # XGBoost search
    param_dist_xgb = {
        "n_estimators":     randint(100, 400),
        "max_depth":        randint(3, 10),
        "learning_rate":    uniform(0.01, 0.2),
        "subsample":        uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "scale_pos_weight": [scale]
    }
    xgb = XGBClassifier(eval_metric="logloss", random_state=42)
    search_xgb = RandomizedSearchCV(xgb, param_dist_xgb, n_iter=20, cv=3,
                                    scoring="recall", random_state=42, n_jobs=-1, verbose=1)
    search_xgb.fit(X_sample, y_sample)
    best_xgb = search_xgb.best_estimator_
    print("Best XGB params:", search_xgb.best_params_)

    # Ensemble
    voting_clf = VotingClassifier(
        estimators=[("rf", best_rf), ("xgb", best_xgb)],
        voting="soft"
    )
    voting_clf.fit(X_train_sm, y_train_sm)

    # Evaluate
    y_prob = voting_clf.predict_proba(X_test)[:, 1]
    best_threshold, threshold_df = find_best_threshold(y_test, y_prob)
    y_final = (y_prob >= best_threshold).astype(int)

    metrics = {
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "recall":    recall_score(y_test, y_final),
        "precision": precision_score(y_test, y_final, zero_division=0),
        "f1":        f1_score(y_test, y_final, zero_division=0),
        "threshold": best_threshold
    }

    print("\n--- Ensemble Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\n", classification_report(y_test, y_final))

    # Log to MLflow
    with mlflow.start_run(run_name="ensemble_rf_xgb"):
        mlflow.log_params(search_rf.best_params_)
        mlflow.log_params({f"xgb_{k}": v for k, v in search_xgb.best_params_.items()})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(voting_clf, artifact_path="model")

        # Save model + threshold locally
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, "voting_clf.pkl"), "wb") as f:
            pickle.dump({"model": voting_clf, "threshold": best_threshold}, f)

        print(f"\nModel saved to {MODEL_DIR}/voting_clf.pkl")
        print(f"MLflow run logged under experiment: {MLFLOW_EXPERIMENT}")

if __name__ == "__main__":
    train()