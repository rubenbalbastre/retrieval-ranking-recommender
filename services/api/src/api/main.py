from __future__ import annotations

from fastapi import FastAPI

from api.ranker import ranker
from api.routers.recommendations import router as recommendations_router

app = FastAPI(title="OpenRecommender API", version="0.1.0")
app.include_router(recommendations_router)


@app.on_event("startup")
def on_startup() -> None:
    ranker.load()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "ranker_loaded": str(ranker.available()).lower()}
