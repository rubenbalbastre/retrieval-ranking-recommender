from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.ranker import ranker


@pytest.mark.unit
def test_health_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(ranker, "load", lambda: None)
    monkeypatch.setattr(ranker, "available", lambda: True)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ranker_loaded": "true"}
