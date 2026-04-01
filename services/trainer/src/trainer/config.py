from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "open_recommender")
    db_user: str = os.getenv("DB_USER", "postgres")
    db_password: str = os.getenv("DB_PASSWORD", "postgres")

    data_dir: Path = Path(os.getenv("DATA_DIR", "data/raw"))
    processed_dir: Path = Path(os.getenv("PROCESSED_DIR", "data/processed"))
    models_dir: Path = Path(os.getenv("MODELS_DIR", "artifacts/models"))

    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "256"))
    content_top_k: int = int(os.getenv("CONTENT_TOP_K", "50"))
    collaborative_top_k: int = int(os.getenv("COLLABORATIVE_TOP_K", "50"))
    final_top_k: int = int(os.getenv("FINAL_TOP_K", "10"))

    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_experiment: str = os.getenv("MLFLOW_EXPERIMENT", "openrecommender-ranking")


settings = Settings()
