FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LOCAL_MODEL_PATH=models/model.pkl

COPY requirements.txt ./
COPY setup.py ./
COPY README.md ./
COPY params.yaml ./
COPY prediction.py ./
COPY src ./src
COPY models ./models

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
