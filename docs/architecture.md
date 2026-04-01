# Architecture

OpenRecommender runs as a local multi-container stack:

- `db`: PostgreSQL + pgvector
- `mlflow`: experiment tracking server
- `trainer`: offline batch pipeline (ingestion, retrieval, ranking, materialization)
- `api`: FastAPI service serving precomputed recommendations

The trainer pipeline is implemented in `services/trainer/src/trainer/pipelines` and writes output to PostgreSQL and `artifacts/models`.
