from __future__ import annotations

import argparse
import re

import pandas as pd

from trainer.config import settings
from trainer.db import execute_sql_file, get_connection


def parse_release_year(title: str) -> int | None:
    m = re.search(r"\((\d{4})\)$", str(title))
    return int(m.group(1)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-schema", action="store_true")
    args = parser.parse_args()

    if args.bootstrap_schema:
        execute_sql_file("infra/db/init/02_schema.sql")

    movies = pd.read_csv(settings.data_dir / "movies.csv")
    ratings = pd.read_csv(settings.data_dir / "ratings.csv")
    tags = pd.read_csv(settings.data_dir / "tags.csv")

    movies["release_year"] = movies["title"].map(parse_release_year)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE ratings, tags, movies RESTART IDENTITY CASCADE;")

            movie_rows = list(movies[["movieId", "title", "genres", "release_year"]].itertuples(index=False, name=None))
            cur.executemany(
                "INSERT INTO movies (movie_id, title, genres, release_year) VALUES (%s, %s, %s, %s)",
                movie_rows,
            )

            rating_rows = list(ratings[["userId", "movieId", "rating", "timestamp"]].itertuples(index=False, name=None))
            cur.executemany(
                "INSERT INTO ratings (user_id, movie_id, rating, ts) VALUES (%s, %s, %s, %s)",
                rating_rows,
            )

            tag_rows = list(tags[["userId", "movieId", "tag", "timestamp"]].itertuples(index=False, name=None))
            cur.executemany(
                "INSERT INTO tags (user_id, movie_id, tag, ts) VALUES (%s, %s, %s, %s)",
                tag_rows,
            )

        conn.commit()

    print("Loaded movies, ratings and tags into PostgreSQL.")


if __name__ == "__main__":
    main()
