FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/raw/ ./data/raw/

RUN mkdir -p data/processed models

RUN python -c "import pandas; print('pandas ok')"
RUN python -c "import sklearn; print('sklearn ok')"
RUN ls data/raw/

ENV DISABLE_MLFLOW=true
RUN python src/preprocess.py 2>&1
RUN python src/train.py 2>&1

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "src.predict:app"]