# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH=/app

# Use pip instead of poetry for simplicity in container
COPY backend/pyproject.toml /app/pyproject.toml

RUN pip install --upgrade pip setuptools wheel \
    && pip install fastapi==0.111.0 uvicorn[standard]==0.30.0 \
    python-dotenv==1.0.1 sqlalchemy==2.0.29 asyncpg==0.29.0 \
    alembic==1.13.1 psycopg2-binary==2.9.9

COPY backend/app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


