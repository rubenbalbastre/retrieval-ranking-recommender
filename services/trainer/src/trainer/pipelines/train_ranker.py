from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
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


def group_sizes(df: pd.DataFrame) -> list[int]:
    return df.groupby("user_id").size().tolist()


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-scores)
    y = labels[order][:k]
    gains = (2**y - 1).astype(float)
    discounts = np.log2(np.arange(2, len(y) + 2))
    dcg = float((gains / discounts).sum())

    ideal = np.sort(labels)[::-1][:k]
    ideal_gains = (2**ideal - 1).astype(float)
    ideal_discounts = np.log2(np.arange(2, len(ideal) + 2))
    idcg = float((ideal_gains / ideal_discounts).sum())
    return 0.0 if idcg == 0 else dcg / idcg


def print_dataset_diagnostics(ds: pd.DataFrame) -> None:
    print("Ranking dataset diagnostics:")
    split_counts = ds["split"].value_counts(dropna=False).to_dict()
    print(f"- rows per split: {split_counts}")

    for split in ["train", "val", "test"]:
        part = ds[ds["split"] == split]
        if part.empty:
            print(f"- {split}: empty")
            continue
        pos_rate = float((part["label"] > 0).mean())
        g = part.groupby("user_id").size()
        positives_users = int(part.groupby("user_id")["label"].max().gt(0).sum())
        print(
            f"- {split}: rows={len(part)}, pos_rate={pos_rate:.4f}, "
            f"group_size[min/med/p95]={int(g.min())}/{float(g.median()):.1f}/{float(g.quantile(0.95)):.1f}, "
            f"users_with_positive={positives_users}/{g.shape[0]}"
        )

    val = ds[ds["split"] == "val"]
    if not val.empty:
        rng = np.random.default_rng(42)
        ndcgs = []
        for _, grp in val.groupby("user_id"):
            y = grp["label"].to_numpy(dtype=np.int32)
            if y.sum() == 0:
                continue
            scores = rng.random(len(grp))
            ndcgs.append(ndcg_at_k(y, scores, k=10))
        if ndcgs:
            print(f"- random baseline ndcg@10 (val): {float(np.mean(ndcgs)):.4f} over {len(ndcgs)} users")
        else:
            print("- random baseline ndcg@10 (val): no users with positives")


def main() -> None:
    with get_connection() as conn:
        ds = pd.DataFrame(
            conn.execute("SELECT user_id, movie_id, label, split, features FROM ranking_dataset").fetchall()
        )

    ds = unpack_features(ds)
    ds["user_id"] = pd.to_numeric(ds["user_id"], errors="coerce")
    ds["movie_id"] = pd.to_numeric(ds["movie_id"], errors="coerce")
    ds["label"] = pd.to_numeric(ds["label"], errors="coerce")
    ds["split"] = ds["split"].astype(str)
    ds = ds.dropna(subset=["user_id", "movie_id", "label"]).copy()
    print_dataset_diagnostics(ds)

    train = ds[ds["split"] == "train"].sort_values(["user_id", "movie_id"])
    val = ds[ds["split"] == "val"].sort_values(["user_id", "movie_id"])

    if train.empty:
        raise RuntimeError("No training rows found in ranking_dataset (split='train').")
    if val.empty:
        raise RuntimeError("No validation rows found in ranking_dataset (split='val').")

    feature_cols = [c for c in train.columns if c not in {"user_id", "movie_id", "label", "split"}]
    for col in feature_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce").fillna(0.0)
        val[col] = pd.to_numeric(val[col], errors="coerce").fillna(0.0)

    dtrain = xgb.DMatrix(train[feature_cols].values, label=train["label"].values)
    dval = xgb.DMatrix(val[feature_cols].values, label=val["label"].values)
    dtrain.set_group(group_sizes(train))
    dval.set_group(group_sizes(val))

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }

    evals_result: dict[str, dict[str, list[float]]] = {}
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run():
        mlflow.log_params(
            {
                **params,
                "num_boost_round": 300,
                "early_stopping_rounds": 30,
                "n_train_rows": int(len(train)),
                "n_val_rows": int(len(val)),
                "n_features": int(len(feature_cols)),
            }
        )

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=25,
            evals_result=evals_result,
        )

        settings.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = Path(settings.models_dir) / "xgb_ranker.json"
        meta_path = Path(settings.models_dir) / "ranker_features.json"

        booster.save_model(str(model_path))
        meta_path.write_text(json.dumps({"feature_cols": feature_cols}, indent=2), encoding="utf-8")

        best_iteration = int(booster.best_iteration)
        best_val_ndcg = float(evals_result["val"]["ndcg@10"][best_iteration])
        mlflow.log_metric("best_val_ndcg_at_10", best_val_ndcg)
        mlflow.log_metric("best_iteration", best_iteration)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(meta_path))

    print(f"Saved model to {model_path} | best_val_ndcg@10={best_val_ndcg:.6f}")


if __name__ == "__main__":
    main()
