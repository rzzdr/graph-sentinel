from __future__ import annotations

from fastapi import APIRouter

from src import __version__
from src.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)  # type: ignore[misc]
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version=__version__,
        mode="simulation",
    )


@router.get("/ready")  # type: ignore[misc]
async def readiness() -> dict[str, str]:
    return {"status": "ready"}
