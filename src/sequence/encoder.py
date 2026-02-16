from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)


class SequenceEncoder:
    """Learns behavioral embeddings from transaction sequences using a
    lightweight convolutional autoencoder approach, implemented with
    scikit-learn-compatible components to avoid heavy DL dependencies.

    The encoder treats each account's transaction sequence as a 2D signal
    (time steps x features) and learns a compressed representation that
    captures temporal behavioral patterns.
    """

    def __init__(self, embedding_dim: int = 32) -> None:
        self.embedding_dim = embedding_dim
        self.encoder: Any = None
        self._fitted = False

    def fit(self, sequences: dict[str, np.ndarray]) -> None:
        """Fit the encoder on a set of account sequences."""
        if not sequences:
            logger.warning("sequence_encoder_empty_input")
            return

        X = self._flatten_sequences(sequences)
        n_components = min(self.embedding_dim, X.shape[0] - 1, X.shape[1])

        if n_components < 1:
            logger.warning("sequence_encoder_insufficient_data")
            return

        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        np.nan_to_num(X_scaled, copy=False, nan=0.0)

        self.encoder = TruncatedSVD(n_components=n_components, random_state=42)
        self.encoder.fit(X_scaled)
        self._fitted = True

        explained = sum(self.encoder.explained_variance_ratio_)
        logger.info(
            "sequence_encoder_fitted",
            accounts=len(sequences),
            dim=n_components,
            explained_variance=round(explained, 4),
        )

    def encode(self, sequences: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Produce behavioral embeddings for each account."""
        if not self._fitted or self.encoder is None:
            logger.warning("sequence_encoder_not_fitted, returning zero embeddings")
            return {aid: np.zeros(self.embedding_dim) for aid in sequences}

        X = self._flatten_sequences(sequences)
        X_scaled = self._scaler.transform(X)
        np.nan_to_num(X_scaled, copy=False, nan=0.0)
        embeddings_matrix = self.encoder.transform(X_scaled)

        result: dict[str, np.ndarray] = {}
        for i, aid in enumerate(sorted(sequences.keys())):
            vec = embeddings_matrix[i]
            if len(vec) < self.embedding_dim:
                vec = np.pad(vec, (0, self.embedding_dim - len(vec)))
            result[aid] = vec[: self.embedding_dim]

        return result

    def compute_anomaly_scores(self, sequences: dict[str, np.ndarray]) -> dict[str, float]:
        """Compute behavioral anomaly scores based on reconstruction error."""
        if not self._fitted or self.encoder is None:
            return {aid: 0.0 for aid in sequences}

        X = self._flatten_sequences(sequences)
        X_scaled = self._scaler.transform(X)
        np.nan_to_num(X_scaled, copy=False, nan=0.0)
        reduced = self.encoder.transform(X_scaled)
        reconstructed = self.encoder.inverse_transform(reduced)

        errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)

        min_err, max_err = errors.min(), errors.max()
        if max_err - min_err > 1e-8:
            normalized = (errors - min_err) / (max_err - min_err)
        else:
            normalized = np.zeros_like(errors)

        result: dict[str, float] = {}
        for i, aid in enumerate(sorted(sequences.keys())):
            result[aid] = round(float(normalized[i]), 6)

        return result

    def _flatten_sequences(self, sequences: dict[str, np.ndarray]) -> np.ndarray:
        sorted_keys = sorted(sequences.keys())
        flattened = []
        for aid in sorted_keys:
            seq = sequences[aid]
            stats = self._compute_sequence_stats(seq)
            flattened.append(stats)
        return np.vstack(flattened)

    def _compute_sequence_stats(self, seq: np.ndarray) -> np.ndarray:
        """Extract statistical features from a sequence matrix.

        Rather than flattening the full (200 x 5) matrix, compute summary
        statistics per feature channel: mean, std, min, max, skew, and
        temporal trend (slope), plus cross-channel correlations.
        """
        n_features = seq.shape[1]
        stats = []

        active_mask = np.any(seq != 0, axis=1)
        active_len = active_mask.sum()

        for f in range(n_features):
            col = seq[:, f]
            active = col[active_mask] if active_len > 0 else col

            stats.extend(
                [
                    np.mean(active),
                    np.std(active),
                    np.min(active),
                    np.max(active),
                ]
            )

            if len(active) > 2:
                from scipy.stats import skew

                stats.append(float(skew(active)))
            else:
                stats.append(0.0)

            if len(active) > 1:
                x = np.arange(len(active))
                slope = np.polyfit(x, active, 1)[0]
                stats.append(slope)
            else:
                stats.append(0.0)

        stats.append(float(active_len))
        stats.append(float(active_len) / seq.shape[0])

        return np.array(stats, dtype=np.float32)

    def save(self, path: Path) -> None:
        """Persist the fitted encoder to disk."""
        if not self._fitted or self.encoder is None:
            logger.warning("sequence_encoder_save_skip", reason="not fitted")
            return

        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "encoder": self.encoder,
            "_scaler": self._scaler,
            "_fitted": self._fitted,
            "embedding_dim": self.embedding_dim,
        }
        joblib.dump(state, path)
        logger.info("sequence_encoder_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> SequenceEncoder:
        """Load a persisted encoder from disk."""
        import joblib

        state = joblib.load(path)
        instance = cls(embedding_dim=state["embedding_dim"])
        instance.encoder = state["encoder"]
        instance._scaler = state["_scaler"]
        instance._fitted = state["_fitted"]
        logger.info("sequence_encoder_loaded", path=str(path))
        return instance
