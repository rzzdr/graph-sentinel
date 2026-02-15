from __future__ import annotations

import json
from pathlib import Path

import click
import structlog

logger = structlog.get_logger(__name__)


@click.group()  # type: ignore[misc]
def cli() -> None:
    """Graph Sentinel: Graph-based mule account detection."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
    )


@cli.command()  # type: ignore[misc]
@click.option("--accounts", default=50_000, help="Number of accounts to generate")  # type: ignore[misc]
@click.option("--days", default=180, help="Simulation duration in days")  # type: ignore[misc]
@click.option("--transactions", default=2_000_000, help="Target transaction count")  # type: ignore[misc]
@click.option("--fraud-ratio", default=0.015, help="Fraud account ratio")  # type: ignore[misc]
@click.option("--seed", default=42, help="Random seed")  # type: ignore[misc]
@click.option("--output-dir", default="data", help="Output directory for generated data")  # type: ignore[misc]
def simulate(
    accounts: int,
    days: int,
    transactions: int,
    fraud_ratio: float,
    seed: int,
    output_dir: str,
) -> None:
    """Generate synthetic financial transaction data with fraud patterns."""
    from src.config.settings import Settings, SimulationConfig
    from src.simulation.orchestrator import SimulationOrchestrator

    settings = Settings(
        data_dir=Path(output_dir),
        simulation=SimulationConfig(
            total_accounts=accounts,
            simulation_days=days,
            total_transactions=transactions,
            fraud_ratio=fraud_ratio,
            seed=seed,
        ),
    )

    orchestrator = SimulationOrchestrator(settings)
    accs, txns = orchestrator.run()

    validation = orchestrator.validate()
    click.echo(f"\nValidation: {json.dumps(validation, indent=2, default=str)}")

    orchestrator.save()
    click.echo(f"\nData saved to {output_dir}/")
    click.echo(f"  Accounts: {accs.height}")
    click.echo(f"  Transactions: {txns.height}")


@cli.command()  # type: ignore[misc]
@click.option("--data-dir", default="data", help="Directory with generated data")  # type: ignore[misc]
@click.option("--output-dir", default="outputs", help="Output directory for results")  # type: ignore[misc]
@click.option(  # type: ignore[misc]
    "--skip-meta-model", is_flag=True, help="Skip meta-model training (use static weights)"
)
@click.option(  # type: ignore[misc]
    "--load-models",
    is_flag=True,
    help="Load previously saved models instead of retraining",
)
@click.option(  # type: ignore[misc]
    "--models-dir",
    default="models",
    help="Directory for saving/loading model weights",
)
def evaluate(
    data_dir: str, output_dir: str, skip_meta_model: bool, load_models: bool, models_dir: str
) -> None:
    """Run the full evaluation pipeline on generated or real data."""
    from src.config.settings import Settings
    from src.features.pipeline import FeatureEngineeringPipeline
    from src.graph.builder import GraphBuilder
    from src.ingestion.pipeline import IngestionPipeline
    from src.monitoring.alerts import AlertService
    from src.scoring.risk_engine import RiskScoringEngine

    settings = Settings(
        data_dir=Path(data_dir), output_dir=Path(output_dir), models_dir=Path(models_dir)
    )
    settings.ensure_dirs()
    model_path = Path(models_dir)

    click.echo("Loading data...")
    pipeline = IngestionPipeline(settings)
    transactions, accounts = pipeline.load_from_parquet()

    click.echo("Building graph...")
    builder = GraphBuilder()
    graph = builder.build_from_transactions(transactions, accounts)
    click.echo(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

    click.echo("Extracting features...")
    feature_pipeline = FeatureEngineeringPipeline()
    features = feature_pipeline.run(graph, transactions, accounts)
    features.write_parquet(Path(output_dir) / "features.parquet")
    click.echo(f"  Features: {features.width - 1}, Accounts: {features.height}")

    propagation_scores = None
    graph_scores = None
    sequence_anomaly_scores = None

    click.echo("Running advanced risk propagation...")
    from src.propagation.engine import RiskPropagationEngine

    prop_engine = RiskPropagationEngine(
        decay_factor=settings.propagation.decay_factor,
        max_hops=settings.propagation.max_hops,
        max_iterations=settings.propagation.max_iterations,
    )
    propagation_scores = prop_engine.propagate(graph)

    click.echo("Computing graph embeddings...")
    from src.graph_learning.embeddings import GraphEmbeddingEngine

    emb_engine = GraphEmbeddingEngine(
        embedding_dim=settings.graph_learning.embedding_dim,
        walk_length=settings.graph_learning.walk_length,
        num_walks=settings.graph_learning.num_walks,
    )
    embeddings = emb_engine.compute_embeddings(graph)

    labels = {
        row["account_id"]: 1 if row["is_fraud"] else 0 for row in accounts.iter_rows(named=True)
    }
    from src.graph_learning.model import GraphClassifier

    graph_clf_path = model_path / "graph_classifier.joblib"
    if load_models and graph_clf_path.exists():
        click.echo("Loading graph classifier from disk...")
        graph_clf = GraphClassifier.load(graph_clf_path)
    else:
        graph_clf = GraphClassifier(embedding_dim=settings.graph_learning.embedding_dim)
        graph_clf.train(embeddings, labels)
        graph_clf.save(graph_clf_path)
    graph_scores = graph_clf.predict(embeddings)

    click.echo("Building behavioral sequences...")
    from src.sequence.builder import SequenceBuilder
    from src.sequence.encoder import SequenceEncoder

    seq_builder = SequenceBuilder(max_seq_len=settings.sequence.max_seq_len)
    sequences = seq_builder.build_sequences(transactions)

    seq_encoder_path = model_path / "sequence_encoder.joblib"
    if load_models and seq_encoder_path.exists():
        click.echo("Loading sequence encoder from disk...")
        seq_encoder = SequenceEncoder.load(seq_encoder_path)
    else:
        seq_encoder = SequenceEncoder(embedding_dim=settings.sequence.embedding_dim)
        seq_encoder.fit(sequences)
        seq_encoder.save(seq_encoder_path)
    sequence_anomaly_scores = seq_encoder.compute_anomaly_scores(sequences)

    click.echo("Computing risk scores...")
    engine = RiskScoringEngine(settings.scoring)

    if not skip_meta_model and settings.scoring.use_meta_model:
        click.echo("Training meta-model...")
        import numpy as np

        from src.scoring.meta_model import MetaScorer

        account_ids = features["account_id"].to_list()
        signal_data: dict[str, list[float]] = {"account_id": account_ids}
        signal_data["sig_behavioral"] = engine._compute_component_score(
            features,
            [
                "incoming_volume",
                "outgoing_volume",
                "net_balance_ratio",
                "velocity_score",
                "burst_index",
                "dormancy_score",
            ],
        )
        signal_data["sig_structural"] = engine._compute_component_score(
            features,
            ["in_degree", "out_degree", "pagerank", "betweenness", "clustering_coefficient"],
        )
        signal_data["sig_fraud_specific"] = engine._compute_component_score(
            features,
            ["circularity_score", "funnel_score", "layering_depth"],
        )
        if propagation_scores:
            signal_data["sig_propagation"] = [
                propagation_scores.get(aid, 0.0) for aid in account_ids
            ]
        if graph_scores:
            signal_data["sig_graph"] = [graph_scores.get(aid, 0.0) for aid in account_ids]
        if sequence_anomaly_scores:
            signal_data["sig_sequence_anomaly"] = [
                sequence_anomaly_scores.get(aid, 0.0) for aid in account_ids
            ]

        import polars as pl

        signals_df = pl.DataFrame(signal_data)
        label_arr = np.array([labels.get(aid, 0) for aid in account_ids])

        meta_scorer_path = model_path / "meta_scorer.joblib"
        if load_models and meta_scorer_path.exists():
            click.echo("Loading meta-scorer from disk...")
            meta_scorer = MetaScorer.load(meta_scorer_path)
        else:
            meta_scorer = MetaScorer()
            meta_result = meta_scorer.fit(signals_df, label_arr)
            click.echo(f"  Meta-model PR-AUC: {meta_result.get('cv_pr_auc', 'N/A')}")
            meta_scorer.save(meta_scorer_path)

        engine.set_meta_scorer(meta_scorer)

        if settings.scoring.use_calibration:
            from src.scoring.calibration import ScoreCalibrator

            calibrator_path = model_path / "calibrator.joblib"
            if load_models and calibrator_path.exists():
                click.echo("Loading calibrator from disk...")
                calibrator = ScoreCalibrator.load(calibrator_path)
            else:
                raw_probs = meta_scorer.predict(signals_df)
                calibrator = ScoreCalibrator()
                cal_result = calibrator.fit(raw_probs, label_arr)
                click.echo(f"  Calibration ECE: {cal_result.get('ece', 'N/A')}")
                calibrator.save(calibrator_path)
            engine.set_calibrator(calibrator)

    risk_scores = engine.score_with_signals(
        features,
        propagation_scores=propagation_scores,
        graph_scores=graph_scores,
        sequence_anomaly_scores=sequence_anomaly_scores,
    )
    risk_scores.write_parquet(Path(output_dir) / "risk_scores.parquet")

    alert_service = AlertService(
        alert_threshold=settings.scoring.alert_threshold,
        high_risk_threshold=settings.scoring.high_risk_threshold,
    )
    alerts = alert_service.generate_alerts(risk_scores)

    click.echo(f"\nResults saved to {output_dir}/")
    click.echo(f"  Total accounts scored: {risk_scores.height}")
    click.echo(f"  Alerts generated: {len(alerts)}")

    import polars as pl

    for level in ["critical", "high", "medium", "low", "minimal"]:
        count = risk_scores.filter(pl.col("risk_level") == level).height
        click.echo(f"  {level}: {count}")


@cli.command()  # type: ignore[misc]
@click.option("--host", default="0.0.0.0", help="Server host")  # type: ignore[misc]
@click.option("--port", default=8000, help="Server port")  # type: ignore[misc]
@click.option("--reload", is_flag=True, help="Enable auto-reload")  # type: ignore[misc]
def serve(host: str, port: int, reload: bool) -> None:
    """Start the FastAPI server."""
    import uvicorn

    uvicorn.run(
        "src.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


def run_simulation() -> None:
    cli(["simulate"])


def run_evaluation() -> None:
    cli(["evaluate"])


def run_server() -> None:
    cli(["serve"])


if __name__ == "__main__":
    cli()
