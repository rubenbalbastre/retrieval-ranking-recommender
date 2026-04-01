FROM python:3.12-slim

WORKDIR /app

COPY services/api/requirements.txt /app/services/api/requirements.txt
RUN pip install --no-cache-dir -r /app/services/api/requirements.txt

COPY . /app

ENV PYTHONPATH=/app/services/api/src

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
