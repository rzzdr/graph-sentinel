from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


class ExplainabilityReporter:
    def generate_account_report(
        self,
        decomposition: dict[str, Any],
        transactions: pl.DataFrame,
        output_dir: Path,
    ) -> Path:
        account_id = decomposition["account_id"]
        report_path = output_dir / f"report_{account_id}.md"
        output_dir.mkdir(parents=True, exist_ok=True)

        account_txns = transactions.filter(
            (pl.col("sender_id") == account_id) | (pl.col("receiver_id") == account_id)
        ).sort("timestamp")

        lines = [
            f"# Risk Report: {account_id}",
            "",
            "## Risk Summary",
            "",
            f"- **Risk Score**: {decomposition.get('risk_score', 'N/A')}",
            f"- **Risk Level**: {decomposition.get('risk_level', 'N/A')}",
            f"- **Behavioral Score**: {decomposition.get('behavioral_risk_score', 'N/A')}",
            f"- **Structural Score**: {decomposition.get('structural_risk_score', 'N/A')}",
            f"- **Network Propagation Score**: "
            f"{decomposition.get('network_propagation_score', 'N/A')}",
            "",
            "## Top Contributing Features",
            "",
            "| Feature | Value | Magnitude |",
            "|---------|-------|-----------|",
        ]

        for feat in decomposition.get("top_contributing_features", []):
            lines.append(
                f"| {feat['feature']} | {feat['value']:.4f} | {feat['abs_magnitude']:.4f} |"
            )

        lines.extend(
            [
                "",
                "## Transaction Summary",
                "",
                f"- Total transactions involving this account: {account_txns.height}",
            ]
        )

        if account_txns.height > 0:
            sent = account_txns.filter(pl.col("sender_id") == account_id)
            received = account_txns.filter(pl.col("receiver_id") == account_id)
            fraud_txns = account_txns.filter(pl.col("is_fraud"))

            lines.extend(
                [
                    f"- Sent: {sent.height}",
                    f"- Received: {received.height}",
                    f"- Fraud-labeled: {fraud_txns.height}",
                    "",
                    "## Recent Transactions (last 20)",
                    "",
                    "| Timestamp | Sender | Receiver | Amount | Type | Fraud |",
                    "|-----------|--------|----------|--------|------|-------|",
                ]
            )

            for row in account_txns.tail(20).iter_rows(named=True):
                lines.append(
                    f"| {row['timestamp']} | {row['sender_id'][:8]}... "
                    f"| {row['receiver_id'][:8]}... | {row['amount']:.2f} "
                    f"| {row['transaction_type']} | {row['is_fraud']} |"
                )

        shap_values = decomposition.get("shap_values")
        if shap_values:
            sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
            lines.extend(
                [
                    "",
                    "## SHAP Values",
                    "",
                    "| Feature | SHAP Value |",
                    "|---------|------------|",
                ]
            )
            for feat_name, value in sorted_shap[:15]:
                lines.append(f"| {feat_name} | {value:.6f} |")

        report_path.write_text("\n".join(lines))
        logger.info("report_generated", path=str(report_path))
        return report_path
