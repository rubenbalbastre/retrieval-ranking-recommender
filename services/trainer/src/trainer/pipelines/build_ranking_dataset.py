from __future__ import annotations

import json

import numpy as np
import pandas as pd

from trainer.db import get_connection


def build_time_split(ratings: pd.DataFrame) -> pd.DataFrame:
    q1 = ratings["ts"].quantile(0.7)
    q2 = ratings["ts"].quantile(0.85)
    ratings["split"] = np.where(ratings["ts"] <= q1, "train", np.where(ratings["ts"] <= q2, "val", "test"))
    return ratings


def main() -> None:
    with get_connection() as conn:
        ratings = pd.read_sql("SELECT user_id, movie_id, rating, ts FROM ratings", conn)
        candidates = pd.read_sql("SELECT * FROM user_candidates", conn)
        movies = pd.read_sql("SELECT movie_id, genres, release_year FROM movies", conn)

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
    label_df["label"] = (label_df["rating"] >= 4.0).astype(int)
    label_df = label_df[["user_id", "movie_id", "split", "label"]]

    joined = joined.merge(label_df, on=["user_id", "movie_id"], how="left")
    joined["label"] = joined["label"].fillna(0).astype(int)
    joined["split"] = joined["split"].fillna("train")

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
