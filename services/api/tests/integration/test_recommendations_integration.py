from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from api.cache import cache_key, get_redis
from api.db import get_connection
from api.main import app


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS", "0") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests against running services.",
)
def test_recommendations_endpoint_smoke() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS", "0") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests against running services.",
)
def test_online_fallback_for_real_train_user() -> None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT user_id
            FROM ranking_dataset
            WHERE split = 'train'
            GROUP BY user_id
            ORDER BY COUNT(*) DESC
            LIMIT 1
            """
        ).fetchone()

    assert row is not None, "No train user found in ranking_dataset."
    user_id = int(row["user_id"])
    limit = 10

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE recommendations
            SET generated_at = NOW() - INTERVAL '10 days'
            WHERE user_id = %s
            """,
            (user_id,),
        )
        conn.commit()

    redis_client = get_redis()
    redis_client.delete(cache_key(user_id, limit))

    with TestClient(app) as client:
        response = client.get(f"/recommendations/{user_id}?limit={limit}")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) > 0
    assert payload[0]["rank"] == 1
    assert redis_client.get(cache_key(user_id, limit)) is not None
