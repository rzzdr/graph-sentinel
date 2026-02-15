from __future__ import annotations

from datetime import datetime

import polars as pl
from fastapi import APIRouter, HTTPException

from src.api.schemas import AlertListResponse, AlertResponse
from src.config.settings import Settings

router = APIRouter()


@router.get("/alerts", response_model=AlertListResponse)  # type: ignore[misc]
async def get_alerts(
    limit: int = 50,
    min_score: float = 75.0,
) -> AlertListResponse:
    limit = min(limit, 1000)
    settings = Settings()
    path = settings.output_dir / "risk_scores.parquet"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Risk scores not available")

    scores = pl.read_parquet(path)
    alerts_df = (
        scores.filter(pl.col("risk_score") >= min_score)
        .sort("risk_score", descending=True)
        .head(limit)
    )

    alerts = []
    for i, row in enumerate(alerts_df.iter_rows(named=True)):
        alerts.append(
            AlertResponse(
                alert_id=i + 1,
                account_id=row["account_id"],
                risk_score=row["risk_score"],
                risk_level=row["risk_level"],
                alert_type="fraud_risk",
                created_at=datetime.now().isoformat(),
                resolved=False,
            )
        )

    return AlertListResponse(alerts=alerts, total=len(alerts))


@router.get("/alerts/stats")  # type: ignore[misc]
async def alert_stats() -> dict[str, object]:
    settings = Settings()
    path = settings.output_dir / "risk_scores.parquet"

    if not path.exists():
        return {"total_alerts": 0, "by_level": {}}

    scores = pl.read_parquet(path)

    thresholds = {"critical": 90.0, "high": 75.0, "medium": 50.0}
    by_level = {}
    prev_threshold = float("inf")
    for level, threshold in thresholds.items():
        count = scores.filter(
            (pl.col("risk_score") >= threshold) & (pl.col("risk_score") < prev_threshold)
        ).height
        by_level[level] = count
        prev_threshold = threshold

    return {
        "total_accounts": scores.height,
        "by_level": by_level,
        "mean_score": round(float(scores["risk_score"].mean()), 2),
    }
