# Architecture

OpenRecommender runs as a local multi-container stack:

- `db`: PostgreSQL + pgvector
- `mlflow`: experiment tracking server
- `trainer`: offline batch pipeline (ingestion, retrieval, ranking, materialization)
- `api`: FastAPI service with cache-aside online scoring and DB fallback
- `redis`: recommendation response cache

The trainer pipeline is implemented in `services/trainer/src/trainer/pipelines` and writes output to PostgreSQL and `artifacts/models`.

## Candidate generation architecture

- Content stage (`pgvector`):
  - set-based SQL with nearest-neighbor retrieval (`<=>`) from `movie_embeddings`
  - excludes seen items per user
  - aggregates and ranks candidates per user in SQL
- Collaborative stage:
  - materializes `item_similarities` from ratings
  - uses positive-only interactions (`rating >= 4`)
  - filters weak pairs (`common_users >= 2`)
  - keeps top 200 neighbors per item
  - generates per-user collaborative candidates in SQL

Both candidate sources are merged in SQL into `user_candidates`.

## Ranking dataset split strategy

- Split is per-user chronological (`train`/`val`/`test`) from interaction timestamps.
- Candidate rows without explicit interaction labels are kept as negatives (`label=0`).
- Unlabeled rows receive deterministic split assignment based on each user's observed split proportions.
- Split coverage is enforced in dataset build step (not in training step).
