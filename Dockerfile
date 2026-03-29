FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/raw/ ./data/raw/

RUN mkdir -p data/processed models

ENV DISABLE_MLFLOW=true
RUN python -c "
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

df = pd.read_csv('data/raw/Loan_default.csv')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('Done reading')
"

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "src.predict:app"]