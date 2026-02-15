from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)

SIGNAL_GROUPS = [
    "supervised",
    "unsupervised",
    "graph",
    "sequence",
    "structural",
    "propagation",
]


class MetaScorer:
    """Learns how to combine heterogeneous risk signals into a single score.

    Instead of hardcoded weights, a stacked generalization model learns the
    optimal blending from labeled data. Each signal group contributes one or
    more columns, and the meta-model learns non-linear interactions between them.
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.signal_columns: list[str] = []
        self._fitted = False

    def fit(
        self,
        signals: pl.DataFrame,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Train the meta-model on stacked signal columns.

        Args:
            signals: DataFrame with account_id + signal columns from all groups.
            labels: Binary fraud labels (0/1) aligned with signals rows.
        """
        self.signal_columns = [
            c
            for c in signals.columns
            if c != "account_id"
            and signals[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        ]

        X = signals.select(self.signal_columns).fill_null(0.0).fill_nan(0.0).to_numpy()
        y = labels.astype(int)

        if len(np.unique(y)) < 2:
            logger.warning("meta_scorer_skip", reason="single class")
            return {"status": "skipped"}  # type: ignore

        from sklearn.utils.class_weight import compute_sample_weight

        sample_weights = compute_sample_weight("balanced", y)

        import lightgbm as lgb

        pos = int(y.sum())
        neg = len(y) - pos

        self.model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            scale_pos_weight=neg / max(pos, 1),
            num_leaves=15,
            min_child_samples=max(5, pos // 5),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )

        from sklearn.metrics import average_precision_score
        from sklearn.model_selection import StratifiedKFold

        cv = StratifiedKFold(n_splits=min(5, max(2, pos)), shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_t, X_v = X[train_idx], X[val_idx]
            y_t, y_v = y[train_idx], y[val_idx]

            self.model.fit(
                X_t,
                y_t,
                sample_weight=sample_weights[train_idx],
                eval_set=[(X_v, y_v)],
            )
            preds = self.model.predict_proba(X_v)[:, 1]
            try:
                scores.append(average_precision_score(y_v, preds))
            except ValueError:
                scores.append(0.0)

        self.model.fit(X, y, sample_weight=sample_weights)
        self._fitted = True

        mean_pr = float(np.mean(scores)) if scores else 0.0
        logger.info(
            "meta_scorer_trained", cv_pr_auc=round(mean_pr, 4), features=len(self.signal_columns)
        )
        return {"cv_pr_auc": mean_pr, "n_features": len(self.signal_columns)}

    def predict(self, signals: pl.DataFrame) -> np.ndarray:
        """Produce raw fraud probabilities from the meta-model."""
        if not self._fitted or self.model is None:
            return np.zeros(signals.height)

        available = [c for c in self.signal_columns if c in signals.columns]
        missing = [c for c in self.signal_columns if c not in signals.columns]

        X = signals.select(available).fill_null(0.0).fill_nan(0.0).to_numpy()

        if missing:
            import warnings

            warnings.warn(
                f"Meta-scorer missing {len(missing)} signal columns, zero-filling", stacklevel=2
            )
            padding = np.zeros((X.shape[0], len(missing)))
            X = np.hstack([X, padding])

        probs = self.model.predict_proba(X)[:, 1]
        return probs

    def get_feature_importance(self) -> dict[str, float]:
        if not self._fitted or self.model is None:
            return {}

        importances = self.model.feature_importances_
        result = {}
        for col, imp in zip(self.signal_columns, importances, strict=False):
            result[col] = float(imp)

        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Path) -> None:
        """Persist the trained meta-scorer to disk."""
        if not self._fitted or self.model is None:
            logger.warning("meta_scorer_save_skip", reason="not trained")
            return

        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "signal_columns": self.signal_columns,
            "_fitted": self._fitted,
        }
        joblib.dump(state, path)
        logger.info("meta_scorer_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> MetaScorer:
        """Load a persisted meta-scorer from disk."""
        import joblib

        state = joblib.load(path)
        instance = cls()
        instance.model = state["model"]
        instance.signal_columns = state["signal_columns"]
        instance._fitted = state["_fitted"]
        logger.info("meta_scorer_loaded", path=str(path))
        return instance
