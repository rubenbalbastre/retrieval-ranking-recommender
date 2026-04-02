# Runbook

## First run

1. Copy env template:
   - `cp .env.example .env`
2. Download MovieLens data:
   - `make data`
3. Start stack:
   - `docker compose up --build`

## Useful URLs

- API health: `http://localhost:8000/health`
- Recommendations: `http://localhost:8000/recommendations/{user_id}`
- MLflow UI: `http://localhost:5000`

## Re-run training

- `make train`

## Notes

- Training and recommendation generation run through the trainer service and stop when trainer exits.
- A warning about Git metadata in MLflow logs is non-blocking in this setup.
