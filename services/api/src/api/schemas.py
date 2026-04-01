from __future__ import annotations

from pydantic import BaseModel


class Recommendation(BaseModel):
    movie_id: int
    score: float
    rank: int
