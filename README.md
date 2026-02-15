# Graph Sentinel

A graph-based mule account detection system with synthetic financial transaction simulation. Built for research and production deployment, it generates realistic transaction networks, injects structured fraud patterns, constructs temporal graphs, engineers features, and scores accounts for mule activity.

## Architecture

The system operates in two modes:

- **Simulation mode** generates synthetic accounts and transactions, injects fraud patterns (funnel, circular, layering, dormant activation, device sharing), and produces labeled datasets for model training.
- **Production mode** ingests real transaction data, builds temporal graphs, computes risk scores, and serves results through a REST API.

```
Transactions ─→ Graph Builder ─→ Feature Pipeline ─→ Risk Engine ─→ API
                     │                  │                  │
                Neo4j Store       ┌─────┴──────┐    ┌─────┴──────┐
                                  │ Graph      │    │ Meta-model │
                                  │ Embeddings │    │ Calibrator │
                                  │ Sequences  │    └────────────┘
                                  │ Propagation│
                                  └────────────┘
                                                    PostgreSQL / Redis
```

### Core Components

| Module | Purpose |
|--------|---------|
| `simulation` | Synthetic account/transaction generation with behavioral profiles |
| `fraud` | Fraud pattern injection engine with 5 pattern types |
| `graph` | Directed weighted graph construction with temporal windows |
| `features` | 36+ features: structural, behavioral, fraud-specific, graph embeddings, sequence embeddings |
| `graph_learning` | Spectral + random-walk node embeddings with XGBoost graph classifier |
| `sequence` | Fixed-length behavioral sequences with TruncatedSVD encoding and anomaly detection |
| `propagation` | Multi-hop iterative belief propagation with hop-decay risk diffusion |
| `scoring` | Meta-model signal blending (LightGBM) with isotonic calibration, fallback to static weights |
| `training` | Supervised (XGBoost, LightGBM, RF) and unsupervised (IsolationForest, LOF, DBSCAN) |
| `explainability` | SHAP-based decomposition and markdown reports |
| `visualization` | Plotly, PyVis, and Matplotlib outputs |
| `api` | FastAPI REST endpoints for scoring, alerts, simulation |
| `monitoring` | Feature drift (KS, PSI, Wasserstein), prediction drift, graph structural drift |

## Requirements

- Python 3.11+
- Poetry
- Docker and Docker Compose (for PostgreSQL, Redis, Neo4j)

## Setup

```bash
# Clone and install
git clone git@github.com:rzzdr/graph-sentinel.git && cd graph-sentinel
poetry install

# Start infrastructure
docker compose up -d

# Verify
poetry run pytest
```

### Environment Variables

All configuration uses the `GS_` prefix. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GS_SIM_TOTAL_ACCOUNTS` | 50000 | Number of synthetic accounts |
| `GS_SIM_TOTAL_TRANSACTIONS` | 2000000 | Number of synthetic transactions |
| `GS_SIM_FRAUD_RATIO` | 0.015 | Fraction of accounts flagged as mules |
| `GS_NEO4J_URI` | bolt://localhost:7687 | Neo4j connection |
| `GS_PG_HOST` | localhost | PostgreSQL host |
| `GS_PG_PORT` | 5432 | PostgreSQL port |
| `GS_REDIS_HOST` | localhost | Redis host |
| `GS_SCORING_ALERT_THRESHOLD` | 75.0 | Risk score threshold for alerts |
| `GS_SCORING_USE_META_MODEL` | true | Enable learned signal blending |
| `GS_SCORING_USE_CALIBRATION` | true | Enable isotonic calibration |
| `GS_PROP_DECAY_FACTOR` | 0.5 | Risk decay per propagation hop |
| `GS_PROP_MAX_HOPS` | 5 | Maximum hops for risk propagation |
| `GS_PROP_MAX_ITERATIONS` | 20 | Maximum propagation iterations |
| `GS_GL_EMBEDDING_DIM` | 64 | Graph embedding dimensionality |
| `GS_GL_WALK_LENGTH` | 40 | Random walk length for embeddings |
| `GS_GL_NUM_WALKS` | 10 | Random walks per node |
| `GS_SEQ_MAX_SEQ_LEN` | 200 | Max transaction sequence length per account |
| `GS_SEQ_EMBEDDING_DIM` | 32 | Sequence embedding dimensionality |

See `src/config/settings.py` for the full list.

## Usage

### CLI Commands

**Run a simulation** — generates accounts, transactions, and injects fraud patterns:

```bash
poetry run gs-simulate --accounts 50000 --days 180 --transactions 2000000
```

**Evaluate** on existing data — builds graph, extracts features, runs propagation, trains the meta-model, calibrates scores, and outputs results:

```bash
poetry run gs-evaluate --data-dir ./data --output-dir ./outputs
```

To skip meta-model training and use static weight aggregation instead:

```bash
poetry run gs-evaluate --data-dir ./data --skip-meta-model
```

To load previously saved model weights instead of retraining:

```bash
poetry run gs-evaluate --data-dir ./data --load-models
```

Models are saved to `models/` by default (gitignored). A custom directory can be specified:

```bash
poetry run gs-evaluate --data-dir ./data --models-dir ./my-models
```

On the first run (or without `--load-models`), the pipeline trains and automatically saves all models. Subsequent runs can pass `--load-models` to skip training and reuse the persisted weights.

**Start the API server**:

```bash
poetry run gs-serve --host 0.0.0.0 --port 8000
```

### API Endpoints

Once the server is running:

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Score a single account
curl http://localhost:8000/api/v1/scoring/ACC_00001

# Batch scoring
curl -X POST http://localhost:8000/api/v1/scoring/batch \
  -H "Content-Type: application/json" \
  -d '{"account_ids": ["ACC_00001", "ACC_00002"]}'

# Top risky accounts
curl http://localhost:8000/api/v1/scoring/top?limit=20

# Score decomposition
curl http://localhost:8000/api/v1/scoring/ACC_00001/decomposition

# List alerts
curl http://localhost:8000/api/v1/alerts?severity=critical

# Trigger simulation
curl -X POST http://localhost:8000/api/v1/simulation/run \
  -H "Content-Type: application/json" \
  -d '{"num_accounts": 5000, "num_transactions": 100000}'
```

### Evaluation Pipeline

The evaluation pipeline follows this sequence:

1. **Data loading** — reads accounts and transactions from Parquet files
2. **Graph construction** — builds directed weighted graphs with temporal windowing (7/30/90-day and lifetime snapshots)
3. **Feature engineering** — computes 36+ node-level features across structural, behavioral, and fraud-specific categories
4. **Risk propagation** — iterative belief propagation from known fraud nodes through the graph, with edge-weight-aware transition matrices and hop-based decay
5. **Graph embeddings** — spectral decomposition and random-walk-based embeddings produce dense node representations; an XGBoost classifier maps embeddings to fraud probabilities
6. **Behavioral sequences** — transaction histories are encoded as fixed-length numerical sequences, compressed via TruncatedSVD, and scored for anomalies via reconstruction error
7. **Meta-model training** — a LightGBM stacking model learns how to blend all signal groups (behavioral, structural, fraud-specific, propagation, graph, sequence) instead of using hardcoded weights
8. **Calibration** — isotonic regression maps raw meta-model probabilities to well-calibrated risk scores, reporting ECE and Brier score
9. **Risk scoring** — final 0–100 risk scores with component breakdown and risk level classification
10. **Alert generation** — flags accounts exceeding configurable risk thresholds

### Feature Categories

**Structural** (from graph topology): in/out degree, PageRank, betweenness centrality, clustering coefficient, k-core number, strongly connected component size, community assignment, ego network density.

**Behavioral** (from transaction patterns): total volume in/out, transaction count, balance ratio, velocity (txn/day), burst index, dormancy ratio, counterparty diversity, repeat transaction ratio.

**Fraud-specific** (designed for mule detection): circularity score, funnel concentration, layering depth, risk propagation (neighbor risk), device cluster size, max chain length.

**Graph embeddings** (from graph learning): spectral embedding components and random-walk co-occurrence embeddings, combined into a configurable-dimension dense vector per node.

**Sequence embeddings** (from behavioral modeling): statistical summaries (mean, std, min, max, skew, slope) per transaction feature channel, compressed via TruncatedSVD.

**Propagation scores** (from risk diffusion): multi-hop propagated risk values from the belief propagation engine, with distance-based decay.

## Project Structure

```
src/
├── config/settings.py            # Pydantic settings for all subsystems
├── simulation/
│   ├── accounts.py               # Account generation
│   ├── transactions.py           # Transaction generation
│   ├── behavioral.py             # Behavioral profile models
│   ├── validator.py              # Statistical validation
│   └── orchestrator.py           # Simulation orchestration
├── fraud/
│   ├── engine.py                 # Fraud injection orchestrator
│   ├── labels.py                 # Ground truth label construction
│   └── patterns/                 # Fraud pattern implementations
│       ├── base.py, funnel.py, circular.py,
│       ├── layering.py, dormant.py, device_sharing.py
├── graph/
│   ├── builder.py                # Graph construction from transactions
│   ├── temporal.py               # Time-aware rolling windows
│   └── neo4j_store.py            # Neo4j persistence
├── graph_learning/
│   ├── embeddings.py             # Spectral + random-walk graph embeddings
│   └── model.py                  # XGBoost classifier over graph embeddings
├── sequence/
│   ├── builder.py                # Transaction-to-sequence conversion
│   └── encoder.py                # TruncatedSVD behavioral encoding + anomaly
├── propagation/
│   └── engine.py                 # Multi-hop belief propagation risk diffusion
├── features/
│   ├── structural.py             # Topological features
│   ├── behavioral.py             # Transaction behavior features
│   ├── fraud_specific.py         # Mule-detection features
│   └── pipeline.py               # Feature engineering pipeline
├── scoring/
│   ├── risk_engine.py            # Risk score computation (static + meta)
│   ├── aggregator.py             # Score normalization & classification
│   ├── meta_model.py             # LightGBM stacking model for signal blending
│   └── calibration.py            # Isotonic regression calibration
├── training/
│   ├── data_split.py             # Temporal train/test splitting
│   ├── supervised.py             # XGBoost, LightGBM, RandomForest
│   ├── unsupervised.py           # IsolationForest, LOF, DBSCAN
│   └── evaluation.py             # Metrics computation
├── explainability/
│   ├── decomposition.py          # SHAP and feature contributions
│   └── reports.py                # Markdown report generation
├── visualization/
│   ├── clusters.py               # Plotly interactive charts
│   ├── graphs.py                 # PyVis network visualizations
│   └── reports.py                # Matplotlib static reports
├── ingestion/
│   ├── schema.py                 # Schema validation (Pydantic)
│   └── pipeline.py               # Data loading pipeline
├── storage/
│   ├── postgres.py               # Async PostgreSQL (SQLAlchemy)
│   └── redis_cache.py            # Redis caching layer
├── api/
│   ├── app.py                    # FastAPI application
│   ├── schemas.py                # Request/response models
│   └── routes/                   # health, scoring, alerts, simulation
├── monitoring/
│   ├── drift.py                  # Feature drift (KS, PSI, Wasserstein)
│   ├── prediction_drift.py       # Score distribution drift detection
│   ├── graph_drift.py            # Graph structural drift monitoring
│   └── alerts.py                 # Alert service
└── cli.py                        # Click CLI (simulate, evaluate, serve)
```

## Tech Stack

| Category | Tools |
|----------|-------|
| Data processing | Polars |
| Graph analytics | NetworkX, Neo4j |
| ML | XGBoost, LightGBM, scikit-learn |
| Explainability | SHAP |
| API | FastAPI, uvicorn |
| Storage | PostgreSQL (asyncpg), Redis |
| Visualization | Plotly, PyVis, Matplotlib |
| Config | Pydantic v2, pydantic-settings |
| Logging | structlog |
| Testing | pytest, pytest-asyncio, httpx |
| Code quality | ruff, mypy, pre-commit |

## Development

```bash
# Run tests
poetry run pytest

# Type checking
poetry run mypy .

# Linting
poetry run ruff check .

# Pre-commit hooks
poetry run pre-commit install
poetry run pre-commit run --all-files
```
