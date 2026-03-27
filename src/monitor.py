import pandas as pd
import numpy as np
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
import mlflow
import os

REFERENCE_PATH = "data/processed/features.csv"
REPORT_DIR = "reports"
MLFLOW_EXPERIMENT = "loan-default-monitoring"

def load_reference_data():
    df = pd.read_csv(REFERENCE_PATH)
    return df.drop(columns=["Default"])

def simulate_production_data(reference_df, drift=True):
    df = reference_df.copy()
    if drift:
        df['LoanToIncome']    = df['LoanToIncome'] * 5.0
        df['RiskInteraction'] = df['RiskInteraction'] * 4.0
        df['CreditScore']     = (df['CreditScore'] * 0.5).clip(300, 850)
        df['DTIRatio']        = df['DTIRatio'] * 3.0
        df['InterestRate']    = df['InterestRate'] * 2.5
    else:
        df['LoanToIncome']    = df['LoanToIncome'] * np.random.uniform(0.95, 1.05, len(df))
        df['RiskInteraction'] = df['RiskInteraction'] * np.random.uniform(0.95, 1.05, len(df))
    return df

def run_monitoring(drift=True):
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    reference_df = load_reference_data()
    ref_sample   = reference_df.sample(n=5000, random_state=42)
    prod_sample  = simulate_production_data(reference_df, drift=drift).sample(n=5000, random_state=42)

    definition   = DataDefinition()
    ref_dataset  = Dataset.from_pandas(ref_sample,  data_definition=definition)
    prod_dataset = Dataset.from_pandas(prod_sample, data_definition=definition)

    report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset(),
    ])

    run = report.run(
        reference_data=ref_dataset,
        current_data=prod_dataset,
    )

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "drift_report.html")
    run.save_html(report_path)
    print(f"Report saved to {report_path}")

    # Extract drift results from flat metrics list
    result          = run.dict()
    metrics         = result.get("metrics", [])
    drift_detected  = False
    drifted_columns = 0
    drift_share     = 0.0

    for metric in metrics:
        if "DriftedColumnsCount" in metric.get("metric_name", ""):
            value           = metric.get("value", {})
            drifted_columns = int(value.get("count", 0))
            drift_share     = float(value.get("share", 0.0))
            drift_detected  = drift_share >= 0.3
            break

    print(f"\n--- Drift Monitoring Results ---")
    print(f"  Drift detected:  {drift_detected}")
    print(f"  Drifted columns: {drifted_columns}")
    print(f"  Drift share:     {drift_share:.2%}")

    with mlflow.start_run(run_name=f"monitor_drift_{drift}"):
        mlflow.log_metric("drift_detected",  int(drift_detected))
        mlflow.log_metric("drifted_columns", drifted_columns)
        mlflow.log_metric("drift_share",     drift_share)
        mlflow.log_artifact(report_path)

    if drift_detected:
        print("\n ALERT: Data drift detected — model retraining recommended.")
    else:
        print("\n No significant drift detected — model is healthy.")

    return drift_detected

if __name__ == "__main__":
    print("=== Simulating DRIFTED production data ===")
    run_monitoring(drift=True)

    print("\n=== Simulating CLEAN production data ===")
    run_monitoring(drift=False)
