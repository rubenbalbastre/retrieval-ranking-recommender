from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.ranker import ranker
from api.routers import recommendations as reco_router


@pytest.mark.unit
def test_recommendations_returns_cached(monkeypatch) -> None:
    cached_rows = [{"movie_id": 10, "score": 0.9, "rank": 1}]

    monkeypatch.setattr(reco_router, "get_cached_recommendations", lambda user_id, limit: cached_rows)
    monkeypatch.setattr(
        reco_router,
        "read_precomputed",
        lambda user_id, limit: (_ for _ in ()).throw(AssertionError("must not query precomputed on cache hit")),
    )

    with TestClient(app) as client:
        response = client.get("/recommendations/1?limit=5")

    assert response.status_code == 200
    assert response.json() == cached_rows


@pytest.mark.unit
def test_recommendations_uses_precomputed_on_cache_miss(monkeypatch) -> None:
    precomputed = [{"movie_id": 20, "score": 0.7, "rank": 1}]
    cache_writes: list[tuple[int, int, list[dict]]] = []

    monkeypatch.setattr(reco_router, "get_cached_recommendations", lambda user_id, limit: None)
    monkeypatch.setattr(reco_router, "read_precomputed", lambda user_id, limit: precomputed)
    monkeypatch.setattr(
        reco_router,
        "read_candidate_features",
        lambda user_id, limit: (_ for _ in ()).throw(AssertionError("must not score online when precomputed exists")),
    )
    monkeypatch.setattr(
        reco_router,
        "set_cached_recommendations",
        lambda user_id, limit, rows: cache_writes.append((user_id, limit, rows)),
    )

    with TestClient(app) as client:
        response = client.get("/recommendations/7?limit=3")

    assert response.status_code == 200
    assert response.json() == precomputed
    assert cache_writes == [(7, 3, precomputed)]


@pytest.mark.unit
def test_recommendations_online_fallback_when_precomputed_missing(monkeypatch) -> None:
    cache_writes: list[tuple[int, int, list[dict]]] = []
    candidates = [
        {"movie_id": 101, "features": {"f1": 0.3}},
        {"movie_id": 102, "features": {"f1": 0.9}},
        {"movie_id": 103, "features": {"f1": 0.1}},
    ]

    monkeypatch.setattr(reco_router, "get_cached_recommendations", lambda user_id, limit: None)
    monkeypatch.setattr(reco_router, "read_precomputed", lambda user_id, limit: [])
    monkeypatch.setattr(reco_router, "read_candidate_features", lambda user_id, limit: candidates)
    monkeypatch.setattr(ranker, "available", lambda: True)
    monkeypatch.setattr(ranker, "score", lambda rows: [0.2, 0.8, 0.1])
    monkeypatch.setattr(
        reco_router,
        "set_cached_recommendations",
        lambda user_id, limit, rows: cache_writes.append((user_id, limit, rows)),
    )

    with TestClient(app) as client:
        response = client.get("/recommendations/9?limit=2")

    assert response.status_code == 200
    assert response.json() == [
        {"movie_id": 102, "score": 0.8, "rank": 1},
        {"movie_id": 101, "score": 0.2, "rank": 2},
    ]
    assert cache_writes == [
        (
            9,
            2,
            [
                {"movie_id": 102, "score": 0.8, "rank": 1},
                {"movie_id": 101, "score": 0.2, "rank": 2},
            ],
        )
    ]
