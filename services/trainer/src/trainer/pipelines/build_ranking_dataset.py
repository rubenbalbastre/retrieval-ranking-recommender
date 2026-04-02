from __future__ import annotations

import json

import numpy as np
import pandas as pd

from trainer.db import get_connection


def build_time_split(ratings: pd.DataFrame) -> pd.DataFrame:
    # Per-user chronological split avoids cross-user temporal leakage artifacts.
    out = ratings.sort_values(["user_id", "ts"]).copy()
    out["rn"] = out.groupby("user_id").cumcount() + 1
    out["n"] = out.groupby("user_id")["movie_id"].transform("count")
    out["p"] = out["rn"] / out["n"].clip(lower=1)
    out["split"] = np.where(out["p"] <= 0.7, "train", np.where(out["p"] <= 0.85, "val", "test"))
    return out.drop(columns=["rn", "n", "p"])


def rating_to_label(r: float) -> int:
    if r >= 5.0:
        return 5
    if r >= 4.5:
        return 4
    if r >= 4.0:
        return 3
    if r >= 3.0:
        return 2
    return 1


def ensure_split_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure ranking dataset has non-empty train and val splits.

    Primary split comes from interaction timestamps. When candidate filtering removes
    all validation rows, we reassign a small user-level holdout from train to val.
    """
    out = df.copy()
    train_count = int((out["split"] == "train").sum())
    val_count = int((out["split"] == "val").sum())

    if train_count == 0:
        raise RuntimeError("No train rows available in ranking dataset after feature build.")

    if val_count == 0:
        train_users = sorted(out.loc[out["split"] == "train", "user_id"].unique().tolist())
        if not train_users:
            raise RuntimeError("No train users available to create validation fallback split.")
        holdout_n = max(1, int(len(train_users) * 0.1))
        holdout_users = set(train_users[-holdout_n:])
        out.loc[(out["split"] == "train") & (out["user_id"].isin(holdout_users)), "split"] = "val"

    return out


def assign_unlabeled_splits(joined: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Assign split to unlabeled candidate rows (no explicit interaction label).

    Strategy:
    - Compute per-user split proportions from observed interactions.
    - Deterministically assign candidate rows by hashing (user_id, movie_id)
      so assignment is stable across runs.
    """
    out = joined.copy()
    split_props = (
        ratings.groupby(["user_id", "split"]).size().unstack(fill_value=0).reindex(columns=["train", "val", "test"], fill_value=0)
    )
    totals = split_props.sum(axis=1).replace(0, 1)
    p_train = (split_props["train"] / totals).to_dict()
    p_val = (split_props["val"] / totals).to_dict()

    missing = out["split"].isna()
    if not missing.any():
        return out

    # Stable hash in [0,1)
    key = out.loc[missing, "user_id"].astype(str) + ":" + out.loc[missing, "movie_id"].astype(str)
    h = pd.util.hash_pandas_object(key, index=False).astype("uint64")
    u = (h % 10_000).astype(float) / 10_000.0

    users = out.loc[missing, "user_id"]
    train_thr = users.map(p_train).fillna(0.7)
    val_thr = train_thr + users.map(p_val).fillna(0.15)

    assigned = np.where(u <= train_thr, "train", np.where(u <= val_thr, "val", "test"))
    out.loc[missing, "split"] = assigned
    return out


def main() -> None:
    with get_connection() as conn:
        ratings = pd.DataFrame(conn.execute("SELECT user_id, movie_id, rating, ts FROM ratings").fetchall())
        candidates = pd.DataFrame(conn.execute("SELECT * FROM user_candidates").fetchall())
        movies = pd.DataFrame(conn.execute("SELECT movie_id, genres, release_year FROM movies").fetchall())

    ratings["user_id"] = pd.to_numeric(ratings["user_id"], errors="coerce")
    ratings["movie_id"] = pd.to_numeric(ratings["movie_id"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings["ts"] = pd.to_numeric(ratings["ts"], errors="coerce")
    ratings = ratings.dropna(subset=["user_id", "movie_id", "rating", "ts"]).copy()

    candidates["user_id"] = pd.to_numeric(candidates["user_id"], errors="coerce")
    candidates["movie_id"] = pd.to_numeric(candidates["movie_id"], errors="coerce")
    candidates = candidates.dropna(subset=["user_id", "movie_id"]).copy()

    movies["movie_id"] = pd.to_numeric(movies["movie_id"], errors="coerce")
    movies["release_year"] = pd.to_numeric(movies["release_year"], errors="coerce")
    movies = movies.dropna(subset=["movie_id"]).copy()

    ratings = build_time_split(ratings)

    user_stats = ratings.groupby("user_id").agg(
        user_num_ratings=("rating", "count"),
        user_avg_rating=("rating", "mean"),
        user_last_ts=("ts", "max"),
    )

    item_stats = ratings.groupby("movie_id").agg(
        item_num_ratings=("rating", "count"),
        item_avg_rating=("rating", "mean"),
    )

    joined = candidates.merge(user_stats, on="user_id", how="left")
    joined = joined.merge(item_stats, on="movie_id", how="left")
    joined = joined.merge(movies, on="movie_id", how="left")

    label_df = ratings[["user_id", "movie_id", "rating", "split"]].copy()
    label_df["label"] = label_df["rating"].map(rating_to_label).astype(int)
    label_df = label_df[["user_id", "movie_id", "split", "label"]]

    joined = joined.merge(label_df, on=["user_id", "movie_id"], how="left")
    joined["label"] = joined["label"].fillna(0).astype(int)
    joined = assign_unlabeled_splits(joined, ratings)
    joined = ensure_split_coverage(joined)

    now_ts = ratings["ts"].max()
    joined["activity_recency_days"] = (now_ts - joined["user_last_ts"]) / (24 * 3600)

    feature_cols = [
        "user_num_ratings",
        "user_avg_rating",
        "activity_recency_days",
        "item_num_ratings",
        "item_avg_rating",
        "release_year",
        "content_score",
        "content_rank",
        "collaborative_score",
        "collaborative_rank",
        "number_of_sources",
    ]

    for col in feature_cols:
        joined[col] = joined[col].fillna(0)

    out_rows = []
    for row in joined.itertuples(index=False):
        features = {col: float(getattr(row, col)) for col in feature_cols}
        out_rows.append((int(row.user_id), int(row.movie_id), int(row.label), str(row.split), json.dumps(features)))

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE ranking_dataset")
            cur.executemany(
                "INSERT INTO ranking_dataset (user_id, movie_id, label, split, features) VALUES (%s, %s, %s, %s, %s::jsonb)",
                out_rows,
            )
        conn.commit()

    print(f"Built ranking dataset with {len(out_rows)} rows.")


if __name__ == "__main__":
    main()
