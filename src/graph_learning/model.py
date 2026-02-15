from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)


class GraphClassifier:
    """Inductive node classifier that operates on graph embeddings + node features.

    Uses a gradient-boosted model over graph embeddings concatenated with tabular
    features, supporting incremental graph updates through re-embedding.
    This avoids a hard PyTorch/DGL dependency while preserving graph-structural
    intelligence in the learned representations.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model: Any = None
        self.feature_columns: list[str] = []

    def train(
        self,
        embeddings: dict[str, np.ndarray],
        labels: dict[str, int],
        node_features: dict[str, np.ndarray] | None = None,
    ) -> dict[str, float]:
        X, y, _nodes = self._prepare_data(embeddings, labels, node_features)

        if len(np.unique(y)) < 2:
            logger.warning("graph_classifier_skip", reason="single class in labels")
            return {"status": "skipped"}  # type: ignore

        import xgboost as xgb
        from sklearn.model_selection import StratifiedKFold
        from sklearn.utils.class_weight import compute_sample_weight

        sample_weights = compute_sample_weight("balanced", y)
        pos = int(y.sum())
        neg = len(y) - pos
        scale_pos_weight = neg / max(pos, 1)

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=6,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        )

        cv = StratifiedKFold(n_splits=min(3, max(2, pos)), shuffle=True, random_state=42)
        val_scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_t, X_v = X[train_idx], X[val_idx]
            y_t, y_v = y[train_idx], y[val_idx]
            sw_t = sample_weights[train_idx]

            self.model.fit(X_t, y_t, sample_weight=sw_t, eval_set=[(X_v, y_v)], verbose=False)

            from sklearn.metrics import average_precision_score

            y_pred = self.model.predict_proba(X_v)[:, 1]
            try:
                val_scores.append(average_precision_score(y_v, y_pred))
            except ValueError:
                val_scores.append(0.0)

        self.model.fit(X, y, sample_weight=sample_weights, verbose=False)

        mean_pr_auc = float(np.mean(val_scores)) if val_scores else 0.0
        logger.info("graph_classifier_trained", cv_pr_auc=round(mean_pr_auc, 4))
        return {"cv_pr_auc": mean_pr_auc}

    def predict(
        self,
        embeddings: dict[str, np.ndarray],
        node_features: dict[str, np.ndarray] | None = None,
    ) -> dict[str, float]:
        if self.model is None:
            return {n: 0.0 for n in embeddings}

        nodes = sorted(embeddings.keys())
        X = self._build_feature_matrix(embeddings, node_features, nodes)
        probs = self.model.predict_proba(X)[:, 1]

        return {node: float(prob) for node, prob in zip(nodes, probs, strict=False)}

    def _prepare_data(
        self,
        embeddings: dict[str, np.ndarray],
        labels: dict[str, int],
        node_features: dict[str, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        common_nodes = sorted(set(embeddings.keys()) & set(labels.keys()))
        X = self._build_feature_matrix(embeddings, node_features, common_nodes)
        y = np.array([labels[n] for n in common_nodes])
        return X, y, common_nodes

    def _build_feature_matrix(
        self,
        embeddings: dict[str, np.ndarray],
        node_features: dict[str, np.ndarray] | None,
        nodes: list[str],
    ) -> np.ndarray:
        emb_list = [embeddings.get(n, np.zeros(self.embedding_dim)) for n in nodes]
        X = np.vstack(emb_list)

        if node_features:
            feat_list = []
            for n in nodes:
                f = node_features.get(n)
                if f is not None:
                    feat_list.append(f)
                else:
                    dim = next(iter(node_features.values())).shape[0] if node_features else 0
                    feat_list.append(np.zeros(dim))

            if feat_list:
                F = np.vstack(feat_list)
                X = np.hstack([X, F])

        return X

    def save(self, path: Path) -> None:
        """Persist the trained classifier to disk."""
        if self.model is None:
            logger.warning("graph_classifier_save_skip", reason="not trained")
            return

        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "embedding_dim": self.embedding_dim,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "feature_columns": self.feature_columns,
        }
        joblib.dump(state, path)
        logger.info("graph_classifier_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> GraphClassifier:
        """Load a persisted classifier from disk."""
        import joblib

        state = joblib.load(path)
        instance = cls(
            embedding_dim=state["embedding_dim"],
            n_estimators=state["n_estimators"],
            learning_rate=state["learning_rate"],
        )
        instance.model = state["model"]
        instance.feature_columns = state["feature_columns"]
        logger.info("graph_classifier_loaded", path=str(path))
        return instance
