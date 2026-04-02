from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiSettings:
    models_dir: str = os.getenv("MODELS_DIR", "artifacts/models")
    model_filename: str = os.getenv("RANKER_MODEL_FILE", "xgb_ranker.json")
    model_features_filename: str = os.getenv("RANKER_FEATURES_FILE", "ranker_features.json")

    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    cache_ttl_seconds: int = int(os.getenv("RECO_CACHE_TTL_SECONDS", "300"))
    precomputed_max_age_seconds: int = int(os.getenv("RECO_PRECOMPUTED_MAX_AGE_SECONDS", "86400"))


settings = ApiSettings()
