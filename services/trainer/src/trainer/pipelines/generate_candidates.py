from __future__ import annotations

import numpy as np
import pandas as pd

from trainer.config import settings
from trainer.db import get_connection


def build_content_candidates(ratings: pd.DataFrame, embeddings: pd.DataFrame) -> pd.DataFrame:
    emb_map = {row.movie_id: np.array(row.embedding, dtype=np.float32) for row in embeddings.itertuples(index=False)}
    all_movie_ids = list(emb_map.keys())
    all_matrix = np.vstack([emb_map[mid] for mid in all_movie_ids])

    rows = []
    liked = ratings[ratings["rating"] >= 4.0]
    for user_id, grp in liked.groupby("user_id"):
        seen = set(grp["movie_id"].tolist())
        source_movies = grp["movie_id"].tolist()
        score_acc: dict[int, float] = {}

        for source_id in source_movies:
            if source_id not in emb_map:
                continue
            sims = all_matrix @ emb_map[source_id]
            top_idx = np.argpartition(-sims, min(settings.content_top_k, len(sims) - 1))[: settings.content_top_k]
            for idx in top_idx:
                movie_id = all_movie_ids[idx]
                if movie_id in seen:
                    continue
                score_acc[movie_id] = max(score_acc.get(movie_id, -1.0), float(sims[idx]))

        ranked = sorted(score_acc.items(), key=lambda x: x[1], reverse=True)[: settings.content_top_k]
        for rank, (movie_id, score) in enumerate(ranked, start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "retrieved_by_content": True,
                    "content_score": score,
                    "content_rank": rank,
                }
            )

    return pd.DataFrame(rows)


def build_collaborative_candidates(ratings: pd.DataFrame) -> pd.DataFrame:
    pivot = ratings.pivot_table(index="user_id", columns="movie_id", values="rating", fill_value=0.0)
    item_mat = pivot.T.values
    norms = np.linalg.norm(item_mat, axis=1, keepdims=True) + 1e-9
    item_norm = item_mat / norms
    sim = item_norm @ item_norm.T

    movie_ids = pivot.columns.to_numpy()
    movie_idx = {m: i for i, m in enumerate(movie_ids)}

    rows = []
    liked = ratings[ratings["rating"] >= 4.0]
    for user_id, grp in liked.groupby("user_id"):
        seen = set(grp["movie_id"].tolist())
        score_acc: dict[int, float] = {}

        for source_id in grp["movie_id"].tolist():
            if source_id not in movie_idx:
                continue
            i = movie_idx[source_id]
            sims = sim[i]
            top_idx = np.argpartition(-sims, min(settings.collaborative_top_k, len(sims) - 1))[: settings.collaborative_top_k]
            for j in top_idx:
                candidate = int(movie_ids[j])
                if candidate in seen:
                    continue
                score_acc[candidate] = max(score_acc.get(candidate, -1.0), float(sims[j]))

        ranked = sorted(score_acc.items(), key=lambda x: x[1], reverse=True)[: settings.collaborative_top_k]
        for rank, (movie_id, score) in enumerate(ranked, start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "retrieved_by_collaborative": True,
                    "collaborative_score": score,
                    "collaborative_rank": rank,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    with get_connection() as conn:
        ratings = pd.read_sql("SELECT user_id, movie_id, rating FROM ratings", conn)
        embeddings = pd.read_sql("SELECT movie_id, embedding FROM movie_embeddings", conn)

    content_df = build_content_candidates(ratings, embeddings)
    collab_df = build_collaborative_candidates(ratings)

    merged = content_df.merge(collab_df, on=["user_id", "movie_id"], how="outer")
    merged["retrieved_by_content"] = merged["retrieved_by_content"].fillna(False)
    merged["retrieved_by_collaborative"] = merged["retrieved_by_collaborative"].fillna(False)
    merged["number_of_sources"] = (
        merged["retrieved_by_content"].astype(int) + merged["retrieved_by_collaborative"].astype(int)
    )

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE user_candidates")
            cur.executemany(
                """
                INSERT INTO user_candidates (
                    user_id, movie_id, retrieved_by_content, retrieved_by_collaborative,
                    content_score, content_rank, collaborative_score, collaborative_rank, number_of_sources
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        int(r.user_id),
                        int(r.movie_id),
                        bool(r.retrieved_by_content),
                        bool(r.retrieved_by_collaborative),
                        None if pd.isna(r.content_score) else float(r.content_score),
                        None if pd.isna(r.content_rank) else int(r.content_rank),
                        None if pd.isna(r.collaborative_score) else float(r.collaborative_score),
                        None if pd.isna(r.collaborative_rank) else int(r.collaborative_rank),
                        int(r.number_of_sources),
                    )
                    for r in merged.itertuples(index=False)
                ],
            )
        conn.commit()

    print(f"Generated {len(merged)} unique candidates.")


if __name__ == "__main__":
    main()
