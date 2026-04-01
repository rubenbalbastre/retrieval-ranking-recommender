# OpenRecommender

Local multi-container recommender system with a two-stage architecture:

1. Candidate generation (content + collaborative)
2. Learning-to-rank with XGBoost
3. Experiment tracking with MLflow
4. Serving via FastAPI

## Repository structure

```text
openrecommender/
  docker-compose.yml
  .env.example
  Makefile
  infra/
    db/
    mlflow/
  services/
    trainer/
    api/
  data/
    raw/
    processed/
  artifacts/
    models/
    mlflow/
  docs/
```

## Services

- `db`: PostgreSQL + pgvector
- `mlflow`: experiment tracking UI/server
- `trainer`: batch pipeline to ingest, build features, train, and materialize recommendations
- `api`: serves `GET /recommendations/{user_id}` from precomputed results

## Quick start

1. Copy env file:
   - `cp .env.example .env`
2. Download MovieLens data:
   - `make data`
3. Build and run:
   - `docker compose up --build`

## URLs

- API health: `http://localhost:8000/health`
- Recommendations: `http://localhost:8000/recommendations/{user_id}`
- MLflow UI: `http://localhost:5000`

## Common commands

- Start in background: `make up`
- Stop: `make down`
- Tail logs: `make logs`
- Download dataset: `make data`
- Re-run trainer pipeline: `make train`

## Trainer entrypoints

- Full pipeline: `python -m trainer.pipelines.run_pipeline`
- Individual stages are in `services/trainer/src/trainer/pipelines/`.
