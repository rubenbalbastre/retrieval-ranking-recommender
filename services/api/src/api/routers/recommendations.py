from __future__ import annotations

import json

from fastapi import APIRouter

from api.cache import get_cached_recommendations, set_cached_recommendations
from api.config import settings
from api.db import get_connection
from api.ranker import ranker
from api.schemas import Recommendation

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


def read_precomputed(user_id: int, limit: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT movie_id, score, rank
            FROM recommendations
            WHERE user_id = %s
              AND generated_at >= NOW() - (INTERVAL '1 second' * %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (user_id, settings.precomputed_max_age_seconds, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def read_candidate_features(user_id: int, limit: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT movie_id, features
            FROM ranking_dataset
            WHERE user_id = %s
            LIMIT %s
            """,
            (user_id, max(limit * 30, 200)),
        ).fetchall()
    out = []
    for r in rows:
        features = r["features"]
        if isinstance(features, str):
            features = json.loads(features)
        out.append({"movie_id": int(r["movie_id"]), "features": dict(features)})
    return out


@router.get("/{user_id}", response_model=list[Recommendation])
def get_recommendations(user_id: int, limit: int = 10) -> list[dict]:
    cached = get_cached_recommendations(user_id, limit)
    if cached is not None:
        return [dict(r) for r in cached]

    result: list[dict]
    precomputed = read_precomputed(user_id, limit)
    if precomputed:
        result = precomputed
    elif ranker.available():
        candidates = read_candidate_features(user_id, limit)
        if candidates:
            scores = ranker.score([c["features"] for c in candidates])
            scored = sorted(
                [
                    {"movie_id": c["movie_id"], "score": float(s)}
                    for c, s in zip(candidates, scores, strict=True)
                ],
                key=lambda x: x["score"],
                reverse=True,
            )[:limit]
            result = [
                {"movie_id": row["movie_id"], "score": row["score"], "rank": i + 1}
                for i, row in enumerate(scored)
            ]
        else:
            result = []
    else:
        result = []

    set_cached_recommendations(user_id, limit, result)
    return result
