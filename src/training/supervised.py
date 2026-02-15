from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import structlog

if TYPE_CHECKING:
    import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

logger = structlog.get_logger(__name__)


class SupervisedTrainer:
    def __init__(self, feature_columns: list[str] | None = None) -> None:
        self.feature_columns = feature_columns
        self.models: dict[str, Any] = {}
        self.best_model_name: str | None = None

    def train(
        self,
        train_data: pl.DataFrame,
        val_data: pl.DataFrame,
    ) -> dict[str, Any]:
        feature_cols = self._resolve_feature_columns(train_data)

        X_train = train_data.select(feature_cols).to_numpy()
        y_train = train_data["is_fraud"].cast(pl.Int32).to_numpy()
        X_val = val_data.select(feature_cols).to_numpy()
        y_val = val_data["is_fraud"].cast(pl.Int32).to_numpy()

        sample_weights = compute_sample_weight("balanced", y_train)

        results: dict[str, Any] = {}

        results["xgboost"] = self._train_xgboost(X_train, y_train, X_val, y_val, sample_weights)
        results["lightgbm"] = self._train_lightgbm(X_train, y_train, X_val, y_val, sample_weights)
        results["random_forest"] = self._train_random_forest(
            X_train, y_train, X_val, y_val, sample_weights
        )

        from src.training.evaluation import ModelEvaluator

        evaluator = ModelEvaluator()
        best_score = -1.0
        for name, model_data in results.items():
            y_prob = model_data["val_predictions"]
            metrics = evaluator.compute_metrics(y_val, y_prob)
            model_data["val_metrics"] = metrics

            pr_auc = metrics.get("pr_auc", 0.0)
            if pr_auc > best_score:
                best_score = pr_auc
                self.best_model_name = name

        logger.info("training_complete", best_model=self.best_model_name, best_pr_auc=best_score)
        return results

    def predict(self, data: pl.DataFrame, model_name: str | None = None) -> np.ndarray:
        name = model_name or self.best_model_name
        if name is None or name not in self.models:
            msg = f"Model '{name}' not found"
            raise ValueError(msg)

        feature_cols = self._resolve_feature_columns(data)
        X = data.select(feature_cols).to_numpy()
        model = self.models[name]

        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    def _resolve_feature_columns(self, data: pl.DataFrame) -> list[str]:
        if self.feature_columns:
            return [c for c in self.feature_columns if c in data.columns]
        exclude = {"account_id", "is_fraud", "fraud_role", "first_seen"}
        return [
            c
            for c in data.columns
            if c not in exclude
            and data[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt32)
        ]

    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray,
    ) -> dict[str, Any]:
        import xgboost as xgb

        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / max(pos_count, 1)

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            early_stopping_rounds=20,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        self.models["xgboost"] = model
        val_pred = model.predict_proba(X_val)[:, 1]

        logger.info("xgboost_trained", best_iteration=model.best_iteration)
        return {"model": model, "val_predictions": val_pred}

    def _train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray,
    ) -> dict[str, Any]:
        import lightgbm as lgb

        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / max(pos_count, 1)

        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            metric="average_precision",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
        )

        self.models["lightgbm"] = model
        val_pred = model.predict_proba(X_val)[:, 1]

        logger.info("lightgbm_trained")
        return {"model": model, "val_predictions": val_pred}

    def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray,
    ) -> dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train, sample_weight=sample_weights)

        self.models["random_forest"] = model
        val_pred = model.predict_proba(X_val)[:, 1]

        logger.info("random_forest_trained")
        return {"model": model, "val_predictions": val_pred}
