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
    vectors = tfidf.fit_transform(corpus).toarray().astype(np.float32)

    if vectors.shape[1] < settings.embedding_dim:
        pad = np.zeros((vectors.shape[0], settings.embedding_dim - vectors.shape[1]), dtype=np.float32)
        vectors = np.hstack([vectors, pad])

    vectors = normalize(vectors, norm="l2")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE movie_embeddings")
            cur.executemany(
                "INSERT INTO movie_embeddings (movie_id, embedding) VALUES (%s, %s)",
                [
                    (int(movie_id), list(map(float, emb.tolist())))
                    for movie_id, emb in zip(df["movie_id"].values, vectors, strict=True)
                ],
            )
        conn.commit()

    print(f"Stored {len(df)} movie embeddings.")


if __name__ == "__main__":
    main()
