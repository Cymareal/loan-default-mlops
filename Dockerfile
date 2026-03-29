FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/raw/ ./data/raw/

RUN mkdir -p data/processed models

ENV DISABLE_MLFLOW=true
RUN python -u src/preprocess.py || true
RUN ls data/processed/
RUN python -u src/train.py || true

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "src.predict:app"]