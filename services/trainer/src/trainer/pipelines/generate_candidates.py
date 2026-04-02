from __future__ import annotations

import numpy as np
import pandas as pd

from trainer.config import settings
from trainer.db import get_connection


def build_content_candidates() -> pd.DataFrame:
    """
    Generate content-based candidates with a set-based SQL pipeline in Postgres.

    SQL workflow:
    1. Build per-user positive seeds (rating >= 4.0).
    2. For each seed, retrieve top-K nearest neighbors via pgvector LATERAL query.
    3. Exclude seen items (any prior user rating).
    4. Aggregate by max(score) for each (user, candidate).
    5. Rank per-user and keep top-K.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            WITH liked AS (
                SELECT DISTINCT user_id, movie_id AS seed_movie_id
                FROM ratings
                WHERE rating >= 4.0
            ),
            seen AS (
                SELECT DISTINCT user_id, movie_id
                FROM ratings
            ),
            neighbors AS (
                SELECT
                    l.user_id,
                    cand.movie_id,
                    1 - (cand.embedding <=> seed.embedding) AS score
                FROM liked AS l
                JOIN movie_embeddings AS seed
                  ON seed.movie_id = l.seed_movie_id
                CROSS JOIN LATERAL (
                    SELECT me.movie_id, me.embedding
                    FROM movie_embeddings AS me
                    WHERE me.movie_id <> l.seed_movie_id
                    ORDER BY seed.embedding <=> me.embedding
                    LIMIT %s
                ) AS cand
            ),
            filtered AS (
                SELECT n.user_id, n.movie_id, n.score
                FROM neighbors AS n
                LEFT JOIN seen AS s
                  ON s.user_id = n.user_id
                 AND s.movie_id = n.movie_id
                WHERE s.movie_id IS NULL
            ),
            aggregated AS (
                SELECT user_id, movie_id, MAX(score) AS content_score
                FROM filtered
                GROUP BY user_id, movie_id
            ),
            ranked AS (
                SELECT
                    user_id,
                    movie_id,
                    content_score,
                    ROW_NUMBER() OVER (
                        PARTITION BY user_id
                        ORDER BY content_score DESC, movie_id
                    ) AS content_rank
                FROM aggregated
            )
            SELECT
                user_id,
                movie_id,
                TRUE AS retrieved_by_content,
                content_score,
                content_rank
            FROM ranked
            WHERE content_rank <= %s
            ORDER BY user_id, content_rank
            """,
            (settings.content_top_k, settings.content_top_k),
        ).fetchall()

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

    content_df = build_content_candidates()
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
