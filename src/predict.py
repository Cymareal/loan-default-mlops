import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

MODEL_PATH = "models/voting_clf.pkl"

FEATURE_ORDER = ['Age', 'Income', 'CreditScore', 'InterestRate', 'LoanTerm',
                 'DTIRatio', 'Education', 'EmploymentType', 'MaritalStatus',
                 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner',
                 'LoanToIncome', 'CreditLinesPerYear', 'RiskInteraction']

def load_model():
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["threshold"]

model, threshold = load_model()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Engineer same features as training
        df['LoanToIncome']       = df['LoanAmount'] / (df['Income'] + 1)
        df['CreditLinesPerYear'] = df['NumCreditLines'] / ((df['MonthsEmployed'] / 12) + 1)
        df['RiskInteraction']    = df['InterestRate'] * df['DTIRatio']

        # Drop redundant columns
        df = df.drop(columns=['NumCreditLines', 'LoanAmount', 'MonthsEmployed'], errors="ignore")

        # Drop target if accidentally included
        df = df.drop(columns=["Default"], errors="ignore")

        # Reorder columns to match training
        df = df[FEATURE_ORDER]

        prob = model.predict_proba(df)[:, 1][0]
        prediction = int(prob >= threshold)

        return jsonify({
            "default_probability": round(float(prob), 4),
            "prediction": prediction,
            "prediction_label": "Default" if prediction == 1 else "No Default",
            "threshold_used": round(float(threshold), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)