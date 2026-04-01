from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from trainer.config import settings
from trainer.db import get_connection


def unpack_features(df: pd.DataFrame) -> pd.DataFrame:
    def _parse(v: object) -> dict:
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            return json.loads(v)
        return {}

    feat = pd.json_normalize(df["features"].map(_parse))
    feat.index = df.index
    return pd.concat([df.drop(columns=["features"]), feat], axis=1)


def main() -> None:
    model_path = Path(settings.models_dir) / "xgb_ranker.json"
    meta_path = Path(settings.models_dir) / "ranker_features.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Model artifacts not found. Run trainer.pipelines.train_ranker first.")

    feature_cols = json.loads(meta_path.read_text(encoding="utf-8"))["feature_cols"]

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    with get_connection() as conn:
        ds = pd.DataFrame(
            conn.execute("SELECT user_id, movie_id, split, features FROM ranking_dataset").fetchall()
        )

    ds = unpack_features(ds)
    ds["user_id"] = pd.to_numeric(ds["user_id"], errors="coerce")
    ds["movie_id"] = pd.to_numeric(ds["movie_id"], errors="coerce")
    ds["split"] = ds["split"].astype(str)
    ds = ds.dropna(subset=["user_id", "movie_id"]).copy()
    infer_df = ds[ds["split"] != "train"].copy()
    if infer_df.empty:
        raise RuntimeError("No non-train rows found to generate recommendations.")

    for col in feature_cols:
        if col not in infer_df.columns:
            infer_df[col] = 0.0
        infer_df[col] = pd.to_numeric(infer_df[col], errors="coerce").fillna(0.0)

    infer_df["score"] = booster.predict(xgb.DMatrix(infer_df[feature_cols].values))
    infer_df = infer_df.sort_values(["user_id", "score"], ascending=[True, False])
    infer_df["rank"] = infer_df.groupby("user_id").cumcount() + 1
    infer_df = infer_df[infer_df["rank"] <= settings.final_top_k]

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE recommendations")
            cur.executemany(
                "INSERT INTO recommendations (user_id, movie_id, score, rank) VALUES (%s, %s, %s, %s)",
                [
                    (int(r.user_id), int(r.movie_id), float(r.score), int(r.rank))
                    for r in infer_df.itertuples(index=False)
                ],
            )
        conn.commit()

    print(f"Stored {len(infer_df)} recommendations.")


if __name__ == "__main__":
    main()
