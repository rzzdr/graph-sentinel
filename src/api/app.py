from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import structlog
from fastapi import FastAPI

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from src.api.routes import alerts, health, scoring, simulation
from src.config.settings import Settings

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("api_startup")
    settings = Settings()
    app.state.settings = settings
    yield
    logger.info("api_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Graph Sentinel",
        description="Graph-based mule account detection API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health.router, tags=["health"])
    app.include_router(scoring.router, prefix="/api/v1", tags=["scoring"])
    app.include_router(alerts.router, prefix="/api/v1", tags=["alerts"])
    app.include_router(simulation.router, prefix="/api/v1", tags=["simulation"])

    return app
