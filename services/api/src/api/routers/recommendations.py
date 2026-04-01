from __future__ import annotations

from fastapi import APIRouter

from api.db import get_connection
from api.schemas import Recommendation

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/{user_id}", response_model=list[Recommendation])
def get_recommendations(user_id: int, limit: int = 10) -> list[Recommendation]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT movie_id, score, rank
            FROM recommendations
            WHERE user_id = %s
            ORDER BY score DESC
            LIMIT %s
            """,
            (user_id, limit),
        ).fetchall()

    return [Recommendation(**row) for row in rows]
