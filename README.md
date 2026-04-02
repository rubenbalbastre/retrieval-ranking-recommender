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

## Recent pipeline design updates

- Candidate generation is now SQL-first in PostgreSQL.
  - Content retrieval uses `pgvector` nearest-neighbor search with set-based SQL.
  - Collaborative retrieval uses materialized `item_similarities` (cosine on co-ratings), also SQL-first.
- Collaborative similarity build has pruning controls:
  - only positive interactions (`rating >= 4.0`)
  - minimum co-raters per pair (`>= 2`)
  - top-N neighbors per item (`200`)
- Ranking dataset split logic moved upstream to dataset building:
  - per-user chronological time split (`train/val/test`)
  - unlabeled candidate rows are kept as negative labels (`label = 0`)
  - unlabeled rows get deterministic split assignment based on per-user split proportions
  - validation coverage is enforced in `build_ranking_dataset.py`
- `train_ranker.py` now expects valid upstream splits (no fallback split policy there).
- MLflow is pinned to `3.10.1` in both trainer and mlflow server images.

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

## Where to review logic

- Candidate generation:
  - `services/trainer/src/trainer/pipelines/generate_candidates.py`
- Ranking dataset construction and split policy:
  - `services/trainer/src/trainer/pipelines/build_ranking_dataset.py`
