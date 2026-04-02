from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from api.config import settings


class Ranker:
    def __init__(self) -> None:
        self.model: xgb.Booster | None = None
        self.feature_cols: list[str] = []

    def load(self) -> None:
        model_path = Path(settings.models_dir) / settings.model_filename
        features_path = Path(settings.models_dir) / settings.model_features_filename
        if not model_path.exists() or not features_path.exists():
            self.model = None
            self.feature_cols = []
            return

        booster = xgb.Booster()
        booster.load_model(str(model_path))
        self.model = booster
        self.feature_cols = json.loads(features_path.read_text(encoding="utf-8"))["feature_cols"]

    def available(self) -> bool:
        return self.model is not None and len(self.feature_cols) > 0

    def score(self, feature_dicts: list[dict]) -> np.ndarray:
        if not self.available() or self.model is None:
            raise RuntimeError("Ranker is not loaded")
        matrix = np.array(
            [[float(row.get(col, 0.0)) for col in self.feature_cols] for row in feature_dicts],
            dtype=np.float32,
        )
        return self.model.predict(xgb.DMatrix(matrix))


ranker = Ranker()
