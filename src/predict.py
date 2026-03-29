import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "voting_clf.pkl")

FEATURE_ORDER = ['Age', 'Income', 'CreditScore', 'InterestRate', 'LoanTerm',
                 'DTIRatio', 'Education', 'EmploymentType', 'MaritalStatus',
                 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner',
                 'LoanToIncome', 'CreditLinesPerYear', 'RiskInteraction']

model = None
threshold = None

def load_model():
    global model, threshold
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    threshold = artifact["threshold"]

def get_model():
    global model, threshold
    if model is None:
        load_model()
    return model, threshold

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        m, t = get_model()
        data = request.get_json()
        df = pd.DataFrame([data])

        df['LoanToIncome']       = df['LoanAmount'] / (df['Income'] + 1)
        df['CreditLinesPerYear'] = df['NumCreditLines'] / ((df['MonthsEmployed'] / 12) + 1)
        df['RiskInteraction']    = df['InterestRate'] * df['DTIRatio']

        df = df.drop(columns=['NumCreditLines', 'LoanAmount', 'MonthsEmployed'], errors="ignore")
        df = df.drop(columns=["Default"], errors="ignore")
        df = df[FEATURE_ORDER]

        prob = m.predict_proba(df)[:, 1][0]
        prediction = int(prob >= t)

        return jsonify({
            "default_probability": round(float(prob), 4),
            "prediction": prediction,
            "prediction_label": "Default" if prediction == 1 else "No Default",
            "threshold_used": round(float(t), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)