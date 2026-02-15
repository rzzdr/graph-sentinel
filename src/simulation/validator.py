from __future__ import annotations

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class SimulationValidator:
    def __init__(self, accounts: pl.DataFrame, transactions: pl.DataFrame) -> None:
        self.accounts = accounts
        self.transactions = transactions

    def validate_all(self) -> dict[str, dict[str, object]]:
        results: dict[str, dict[str, object]] = {}
        results["amount_distribution"] = self._validate_amount_distribution()
        results["degree_distribution"] = self._validate_degree_distribution()
        results["temporal_coverage"] = self._validate_temporal_coverage()
        results["fraud_ratio"] = self._validate_fraud_ratio()
        results["account_coverage"] = self._validate_account_coverage()

        all_passed = all(r.get("passed", False) for r in results.values())
        logger.info("validation_complete", all_passed=all_passed, results=results)
        return results

    def _validate_amount_distribution(self) -> dict[str, object]:
        """Validate that transaction amounts follow approximately log-normal distribution."""
        amounts = self.transactions["amount"].to_numpy()
        log_amounts = np.log(amounts[amounts > 0])

        from scipy import stats

        if len(log_amounts) < 8:
            return {
                "passed": False,
                "reason": f"insufficient positive-amount transactions ({len(log_amounts)})",
            }

        _, p_value = stats.normaltest(log_amounts[:10000])
        mean = float(np.mean(log_amounts))
        std = float(np.std(log_amounts))

        passed = std > 0.5 and mean > 0
        return {
            "passed": passed,
            "log_mean": round(mean, 4),
            "log_std": round(std, 4),
            "normality_p_value": round(float(p_value), 6),
            "sample_size": len(log_amounts),
        }

    def _validate_degree_distribution(self) -> dict[str, object]:
        """Validate that degree distribution approximates power-law."""
        out_degrees = (
            self.transactions.group_by("sender_id")
            .agg(pl.len().alias("out_degree"))["out_degree"]
            .to_numpy()
        )

        in_degrees = (
            self.transactions.group_by("receiver_id")
            .agg(pl.len().alias("in_degree"))["in_degree"]
            .to_numpy()
        )

        total_degrees = np.concatenate([out_degrees, in_degrees])

        try:
            import powerlaw

            fit = powerlaw.Fit(total_degrees, discrete=True, verbose=False)
            alpha = float(fit.alpha)
            xmin = float(fit.xmin)
            passed = 1.5 < alpha < 4.0
        except Exception:
            alpha = -1.0
            xmin = -1.0
            passed = False

        return {
            "passed": passed,
            "power_law_alpha": round(alpha, 4),
            "power_law_xmin": round(xmin, 4),
            "mean_degree": round(float(np.mean(total_degrees)), 2),
            "max_degree": int(np.max(total_degrees)),
        }

    def _validate_temporal_coverage(self) -> dict[str, object]:
        """Validate transactions span the simulation period."""
        ts = self.transactions["timestamp"]
        min_ts = ts.min()
        max_ts = ts.max()

        if min_ts is None or max_ts is None:
            return {"passed": False, "reason": "no timestamps"}

        span_days = (max_ts - min_ts).days
        passed = span_days > 30

        daily_counts = (
            self.transactions.with_columns(pl.col("timestamp").dt.date().alias("date"))
            .group_by("date")
            .agg(pl.len().alias("count"))
        )

        return {
            "passed": passed,
            "span_days": span_days,
            "active_days": daily_counts.height,
            "avg_daily_txns": round(float(daily_counts["count"].mean()), 1),
        }

    def _validate_fraud_ratio(self) -> dict[str, object]:
        fraud_count = self.accounts.filter(pl.col("is_fraud")).height
        total = self.accounts.height
        ratio = fraud_count / total if total > 0 else 0

        return {
            "passed": 0.001 <= ratio <= 0.05,
            "fraud_accounts": fraud_count,
            "total_accounts": total,
            "ratio": round(ratio, 6),
        }

    def _validate_account_coverage(self) -> dict[str, object]:
        """Check what fraction of accounts participate in transactions."""
        active_senders = self.transactions["sender_id"].n_unique()
        active_receivers = self.transactions["receiver_id"].n_unique()
        all_active = pl.concat(
            [
                self.transactions["sender_id"],
                self.transactions["receiver_id"],
            ]
        ).n_unique()

        total = self.accounts.height
        coverage = all_active / total if total > 0 else 0

        return {
            "passed": coverage > 0.5,
            "active_senders": active_senders,
            "active_receivers": active_receivers,
            "unique_active_accounts": all_active,
            "total_accounts": total,
            "coverage_ratio": round(coverage, 4),
        }
