from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from trainer.config import settings
from trainer.db import get_connection


def main() -> None:
    with get_connection() as conn:
        movies = pd.read_sql("SELECT movie_id, title, genres FROM movies", conn)
        tags = pd.read_sql(
            "SELECT movie_id, string_agg(tag, ' ') AS tags_text FROM tags GROUP BY movie_id",
            conn,
        )

    df = movies.merge(tags, on="movie_id", how="left")
    df["tags_text"] = df["tags_text"].fillna("")
    corpus = (df["title"].fillna("") + " " + df["genres"].fillna("") + " " + df["tags_text"]).tolist()

    tfidf = TfidfVectorizer(max_features=settings.embedding_dim)
    vectors = tfidf.fit_transform(corpus)
    vectors = normalize(vectors, norm="l2")

    current_dim = vectors.shape[1]
    target_dim = settings.embedding_dim
    pad_width = max(0, target_dim - current_dim)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE movie_embeddings")
            batch: list[tuple[int, list[float]]] = []
            for idx, movie_id in enumerate(df["movie_id"].values):
                emb = vectors.getrow(idx).toarray().ravel().astype(np.float32)
                if pad_width > 0:
                    emb = np.pad(emb, (0, pad_width), mode="constant")
                batch.append((int(movie_id), list(map(float, emb.tolist()))))
                if len(batch) >= 500:
                    cur.executemany(
                        "INSERT INTO movie_embeddings (movie_id, embedding) VALUES (%s, %s)",
                        batch,
                    )
                    batch.clear()
            if batch:
                cur.executemany(
                    "INSERT INTO movie_embeddings (movie_id, embedding) VALUES (%s, %s)",
                    batch,
                )
        conn.commit()

    print(f"Stored {len(df)} movie embeddings.")


if __name__ == "__main__":
    main()
