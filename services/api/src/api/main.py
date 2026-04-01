from __future__ import annotations

from fastapi import FastAPI

from api.routers.recommendations import router as recommendations_router

app = FastAPI(title="OpenRecommender API", version="0.1.0")
app.include_router(recommendations_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
