FROM python:3.12-slim

WORKDIR /app

COPY services/trainer/requirements.txt /app/services/trainer/requirements.txt
RUN pip install --no-cache-dir -r /app/services/trainer/requirements.txt

COPY . /app

ENV PYTHONPATH=/app/services/trainer/src
