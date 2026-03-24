import pytest
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"

def test_predict_default(client):
    payload = {
        "Age": 35, "Income": 50000, "CreditScore": 650,
        "LoanAmount": 15000, "LoanTerm": 36, "InterestRate": 12.5,
        "DTIRatio": 0.4, "NumCreditLines": 3, "MonthsEmployed": 24,
        "Education": 1, "EmploymentType": 1, "MaritalStatus": 1,
        "HasMortgage": 0, "HasDependents": 1, "LoanPurpose": 1, "HasCoSigner": 0
    }
    response = client.post("/predict",
                           data=json.dumps(payload),
                           content_type="application/json")
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert "default_probability" in data
    assert "prediction_label" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["default_probability"] <= 1.0

def test_predict_missing_fields(client):
    payload = {"Age": 35, "Income": 50000}
    response = client.post("/predict",
                           data=json.dumps(payload),
                           content_type="application/json")
    assert response.status_code == 400
    assert "error" in response.get_json()