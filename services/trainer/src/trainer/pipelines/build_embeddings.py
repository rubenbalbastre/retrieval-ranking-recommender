from __future__ import annotations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from trainer.config import settings
from trainer.db import get_connection


def main() -> None:
    with get_connection() as conn:
        movies = pd.DataFrame(
            conn.execute("SELECT movie_id, title, genres FROM movies").fetchall()
        )
        tags = pd.DataFrame(
            conn.execute(
                "SELECT movie_id, string_agg(tag, ' ') AS tags_text FROM tags GROUP BY movie_id"
            ).fetchall()
        )

    df = movies.merge(tags, on="movie_id", how="left")
    df["tags_text"] = df["tags_text"].fillna("")
    corpus = (df["title"].fillna("") + " " + df["genres"].fillna("") + " " + df["tags_text"]).tolist()
    model = SentenceTransformer(settings.embedding_model)
    vectors = model.encode(
        corpus,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=256,
    ).astype(np.float32)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE movie_embeddings")
            batch: list[tuple[int, list[float]]] = []
            for idx, movie_id in enumerate(df["movie_id"].values):
                emb = vectors[idx]
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
