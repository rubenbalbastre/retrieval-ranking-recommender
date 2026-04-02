from __future__ import annotations

from trainer.config import settings
from trainer.db import get_connection


def materialize_content_candidates(conn) -> None:
    conn.execute("DROP TABLE IF EXISTS tmp_content_candidates")
    conn.execute(
        """
        CREATE TEMP TABLE tmp_content_candidates AS
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
        )
        SELECT
            user_id,
            movie_id,
            content_score,
            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY content_score DESC, movie_id
            ) AS content_rank
        FROM aggregated
        """,
        (settings.content_top_k,),
    )


def materialize_collaborative_candidates(conn) -> None:
    min_common_users = 2
    max_neighbors_per_item = 200

    conn.execute("TRUNCATE TABLE item_similarities")
    conn.execute(
        """
        WITH pairs AS (
            SELECT
                r1.movie_id AS item_id,
                r2.movie_id AS similar_item_id,
                COUNT(*) AS common_users,
                SUM(r1.rating * r2.rating) AS dot,
                SQRT(SUM(r1.rating * r1.rating)) AS norm_i,
                SQRT(SUM(r2.rating * r2.rating)) AS norm_j
            FROM ratings AS r1
            JOIN ratings AS r2
              ON r1.user_id = r2.user_id
             AND r1.movie_id <> r2.movie_id
            WHERE r1.rating >= 4.0
              AND r2.rating >= 4.0
            GROUP BY r1.movie_id, r2.movie_id
            HAVING COUNT(*) >= %s
        ),
        scored AS (
            SELECT
                item_id,
                similar_item_id,
                CASE
                    WHEN norm_i > 0 AND norm_j > 0 THEN dot / (norm_i * norm_j)
                    ELSE 0.0
                END AS similarity
            FROM pairs
        ),
        ranked AS (
            SELECT
                item_id,
                similar_item_id,
                similarity,
                ROW_NUMBER() OVER (
                    PARTITION BY item_id
                    ORDER BY similarity DESC, similar_item_id
                ) AS rnk
            FROM scored
        )
        INSERT INTO item_similarities (item_id, similar_item_id, similarity)
        SELECT
            item_id,
            similar_item_id,
            similarity
        FROM ranked
        WHERE rnk <= %s
        """,
        (min_common_users, max_neighbors_per_item),
    )

    conn.execute("DROP TABLE IF EXISTS tmp_collab_candidates")
    conn.execute(
        """
        CREATE TEMP TABLE tmp_collab_candidates AS
        WITH liked AS (
            SELECT DISTINCT user_id, movie_id AS seed_movie_id
            FROM ratings
            WHERE rating >= 4.0
        ),
        seen AS (
            SELECT DISTINCT user_id, movie_id
            FROM ratings
        ),
        seed_neighbors AS (
            SELECT
                l.user_id,
                s.similar_item_id AS movie_id,
                s.similarity AS score
            FROM liked AS l
            JOIN item_similarities AS s
              ON s.item_id = l.seed_movie_id
        ),
        filtered AS (
            SELECT sn.user_id, sn.movie_id, sn.score
            FROM seed_neighbors AS sn
            LEFT JOIN seen AS sv
              ON sv.user_id = sn.user_id
             AND sv.movie_id = sn.movie_id
            WHERE sv.movie_id IS NULL
        ),
        aggregated AS (
            SELECT user_id, movie_id, MAX(score) AS collaborative_score
            FROM filtered
            GROUP BY user_id, movie_id
        )
        SELECT
            user_id,
            movie_id,
            collaborative_score,
            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY collaborative_score DESC, movie_id
            ) AS collaborative_rank
        FROM aggregated
        """
    )


def main() -> None:
    with get_connection() as conn:
        materialize_content_candidates(conn)
        materialize_collaborative_candidates(conn)

        conn.execute("TRUNCATE TABLE user_candidates")
        conn.execute(
            """
            INSERT INTO user_candidates (
                user_id,
                movie_id,
                retrieved_by_content,
                retrieved_by_collaborative,
                content_score,
                content_rank,
                collaborative_score,
                collaborative_rank,
                number_of_sources
            )
            SELECT
                COALESCE(c.user_id, k.user_id) AS user_id,
                COALESCE(c.movie_id, k.movie_id) AS movie_id,
                (c.user_id IS NOT NULL) AS retrieved_by_content,
                (k.user_id IS NOT NULL) AS retrieved_by_collaborative,
                c.content_score,
                c.content_rank,
                k.collaborative_score,
                k.collaborative_rank,
                ((c.user_id IS NOT NULL)::int + (k.user_id IS NOT NULL)::int) AS number_of_sources
            FROM (
                SELECT *
                FROM tmp_content_candidates
                WHERE content_rank <= %s
            ) AS c
            FULL OUTER JOIN (
                SELECT *
                FROM tmp_collab_candidates
                WHERE collaborative_rank <= %s
            ) AS k
              ON c.user_id = k.user_id
             AND c.movie_id = k.movie_id
            """,
            (settings.content_top_k, settings.collaborative_top_k),
        )

        count = conn.execute("SELECT COUNT(*) AS n FROM user_candidates").fetchone()["n"]
        conn.commit()

    print(f"Generated {count} unique candidates.")


if __name__ == "__main__":
    main()
