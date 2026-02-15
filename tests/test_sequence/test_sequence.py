from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl

from src.sequence.builder import SequenceBuilder
from src.sequence.encoder import SequenceEncoder


class TestSequenceBuilder:
    def test_build_sequences(self, sample_transactions: pl.DataFrame) -> None:
        builder = SequenceBuilder(max_seq_len=50)
        sequences = builder.build_sequences(sample_transactions)

        assert isinstance(sequences, dict)
        assert len(sequences) > 0

    def test_sequence_shape(self, sample_transactions: pl.DataFrame) -> None:
        max_len = 50
        builder = SequenceBuilder(max_seq_len=max_len)
        sequences = builder.build_sequences(sample_transactions)

        for aid, seq in sequences.items():
            assert seq.shape == (max_len, 5), f"Account {aid} has shape {seq.shape}"
            assert seq.dtype == np.float32

    def test_amount_is_normalized(self, sample_transactions: pl.DataFrame) -> None:
        builder = SequenceBuilder(max_seq_len=50)
        sequences = builder.build_sequences(sample_transactions)

        for seq in sequences.values():
            amounts = seq[:, 0]
            assert amounts.min() >= 0.0
            assert amounts.max() <= 1.0 + 1e-6

    def test_temporal_features_normalized(self, sample_transactions: pl.DataFrame) -> None:
        builder = SequenceBuilder(max_seq_len=50)
        sequences = builder.build_sequences(sample_transactions)

        for seq in sequences.values():
            hour = seq[:, 2]
            dow = seq[:, 3]
            assert hour.min() >= 0.0
            assert hour.max() <= 1.0 + 1e-6
            assert dow.min() >= 0.0
            assert dow.max() <= 1.0 + 1e-6

    def test_includes_both_sender_and_receiver(self, sample_transactions: pl.DataFrame) -> None:
        builder = SequenceBuilder(max_seq_len=200)
        sequences = builder.build_sequences(sample_transactions)

        senders = set(sample_transactions["sender_id"].unique().to_list())
        receivers = set(sample_transactions["receiver_id"].unique().to_list())
        all_accounts = senders | receivers

        assert set(sequences.keys()) == all_accounts


class TestSequenceEncoder:
    def _make_sequences(self, n: int = 20, max_len: int = 50) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(42)
        return {f"acc_{i}": rng.random((max_len, 5)).astype(np.float32) for i in range(n)}

    def test_fit_and_encode(self) -> None:
        sequences = self._make_sequences()
        encoder = SequenceEncoder(embedding_dim=8)
        encoder.fit(sequences)

        embeddings = encoder.encode(sequences)
        assert len(embeddings) == len(sequences)
        for vec in embeddings.values():
            assert vec.shape == (8,)

    def test_anomaly_scores(self) -> None:
        sequences = self._make_sequences()
        encoder = SequenceEncoder(embedding_dim=8)
        encoder.fit(sequences)

        scores = encoder.compute_anomaly_scores(sequences)
        assert len(scores) == len(sequences)
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_encode_before_fit_returns_zeros(self) -> None:
        sequences = self._make_sequences(5)
        encoder = SequenceEncoder(embedding_dim=8)
        embeddings = encoder.encode(sequences)

        for vec in embeddings.values():
            assert np.all(vec == 0.0)

    def test_anomaly_before_fit_returns_zeros(self) -> None:
        sequences = self._make_sequences(5)
        encoder = SequenceEncoder(embedding_dim=8)
        scores = encoder.compute_anomaly_scores(sequences)

        for score in scores.values():
            assert score == 0.0

    def test_empty_input(self) -> None:
        encoder = SequenceEncoder(embedding_dim=8)
        encoder.fit({})
        assert not encoder._fitted

    def test_with_real_transactions(self, sample_transactions: pl.DataFrame) -> None:
        builder = SequenceBuilder(max_seq_len=50)
        sequences = builder.build_sequences(sample_transactions)

        encoder = SequenceEncoder(embedding_dim=8)
        encoder.fit(sequences)

        embeddings = encoder.encode(sequences)
        assert len(embeddings) == len(sequences)

        scores = encoder.compute_anomaly_scores(sequences)
        assert len(scores) == len(sequences)
