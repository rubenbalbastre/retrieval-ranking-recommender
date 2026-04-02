from __future__ import annotations

import numpy as np
import pandas as pd

from trainer.config import settings
from trainer.db import get_connection


def parse_embedding(value: object) -> np.ndarray:
    if isinstance(value, str):
        cleaned = value.strip().strip("[]")
        if not cleaned:
            return np.array([], dtype=np.float32)
        return np.fromstring(cleaned, sep=",", dtype=np.float32)
    return np.array(value, dtype=np.float32)


def build_content_candidates(ratings: pd.DataFrame, embeddings: pd.DataFrame) -> pd.DataFrame:
    """
    Generate content-based candidates per user using embedding similarity.

    Expected inputs:
    - ratings columns: user_id, movie_id, rating
    - embeddings columns: movie_id, embedding

    Logic:
    1. Build a dense embedding matrix for all movies.
    2. For each user, take positively-rated movies (rating >= 4) as retrieval seeds.
    3. Compute similarities between all movies and that user's seed movies in one matrix op.
    4. Aggregate per-movie similarity with max() across seeds.
    5. Exclude already seen items for that user and keep top-K by similarity.

    Notes on terminology:
    - liked_movie_ids: seed set used to retrieve neighbors (positive feedback).
    - seen: items excluded from recommendation output.
      In the current implementation, seen matches liked items. In stricter setups,
      seen can be all historically interacted items (any rating), while seeds remain
      only positive items.
    """
    emb_map = {int(row.movie_id): parse_embedding(row.embedding) for row in embeddings.itertuples(index=False)}
    all_movie_ids = list(emb_map.keys())
    all_matrix = np.vstack([emb_map[mid] for mid in all_movie_ids]) # [n_movies, dim]
    movie_index = {movie_id: idx for idx, movie_id in enumerate(all_movie_ids)}

    rows = []
    liked = ratings[ratings["rating"] >= 4.0]
    for user_id, grp in liked.groupby("user_id"):
        seen = set(grp["movie_id"].tolist())
        liked_movie_ids = [int(mid) for mid in grp["movie_id"].tolist() if int(mid) in movie_index]
        if not liked_movie_ids:
            continue

        source_idx = [movie_index[mid] for mid in liked_movie_ids]
        source_matrix = all_matrix[source_idx]  # [n_source, dim]
        sims = all_matrix @ source_matrix.T  # [n_movies, n_source]
        agg_scores = sims.max(axis=1)  # max similarity against liked movies

        if seen:
            seen_idx = np.array([movie_index[mid] for mid in seen if mid in movie_index], dtype=np.int64)
            agg_scores[seen_idx] = -np.inf

        k = min(settings.content_top_k, len(agg_scores))
        if k <= 0:
            continue
        top_idx = np.argpartition(-agg_scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-agg_scores[top_idx])]
        ranked = [(all_movie_ids[int(idx)], float(agg_scores[int(idx)])) for idx in top_idx]

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
        ratings = pd.DataFrame(conn.execute("SELECT user_id, movie_id, rating FROM ratings").fetchall())
        embeddings = pd.DataFrame(conn.execute("SELECT movie_id, embedding FROM movie_embeddings").fetchall())

    content_df = build_content_candidates(ratings, embeddings)
    collab_df = build_collaborative_candidates(ratings)

    merged = content_df.merge(collab_df, on=["user_id", "movie_id"], how="outer")
    merged["retrieved_by_content"] = merged["retrieved_by_content"].eq(True)
    merged["retrieved_by_collaborative"] = merged["retrieved_by_collaborative"].eq(True)
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
