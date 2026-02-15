from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class AlertService:
    def __init__(self, alert_threshold: float = 75.0, high_risk_threshold: float = 90.0) -> None:
        self.alert_threshold = alert_threshold
        self.high_risk_threshold = high_risk_threshold

    def generate_alerts(self, risk_scores: pl.DataFrame) -> list[dict[str, Any]]:
        flagged = risk_scores.filter(pl.col("risk_score") >= self.alert_threshold)
        flagged = flagged.sort("risk_score", descending=True)

        alerts: list[dict[str, Any]] = []
        for row in flagged.iter_rows(named=True):
            severity = "critical" if row["risk_score"] >= self.high_risk_threshold else "high"

            alerts.append(
                {
                    "account_id": row["account_id"],
                    "risk_score": row["risk_score"],
                    "risk_level": row["risk_level"],
                    "severity": severity,
                    "alert_type": self._determine_alert_type(row),
                    "created_at": datetime.now().isoformat(),
                    "resolved": False,
                }
            )

        logger.info(
            "alerts_generated",
            total=len(alerts),
            critical=sum(1 for a in alerts if a["severity"] == "critical"),
        )
        return alerts

    def _determine_alert_type(self, score_row: dict[str, Any]) -> str:
        behavioral = score_row.get("behavioral_risk_score", 0)
        structural = score_row.get("structural_risk_score", 0)
        propagation = score_row.get("network_propagation_score", 0)

        max_component = max(behavioral, structural, propagation)

        if max_component == propagation:
            return "network_risk"
        if max_component == structural:
            return "structural_anomaly"
        return "behavioral_anomaly"

    def prioritize_alerts(
        self,
        alerts: list[dict[str, Any]],
        max_daily_alerts: int = 100,
    ) -> list[dict[str, Any]]:
        sorted_alerts = sorted(alerts, key=lambda a: a["risk_score"], reverse=True)
        return sorted_alerts[:max_daily_alerts]
