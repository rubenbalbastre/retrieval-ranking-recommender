from __future__ import annotations

import json

import redis

from api.config import settings


def get_redis() -> redis.Redis:
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True,
    )


def cache_key(user_id: int, limit: int) -> str:
    return f"reco:v1:user:{user_id}:limit:{limit}"


def get_cached_recommendations(user_id: int, limit: int) -> list[dict] | None:
    payload = get_redis().get(cache_key(user_id, limit))
    if not payload:
        return None
    return json.loads(payload)


def set_cached_recommendations(user_id: int, limit: int, rows: list[dict]) -> None:
    get_redis().setex(
        cache_key(user_id, limit),
        settings.cache_ttl_seconds,
        json.dumps(rows),
    )
