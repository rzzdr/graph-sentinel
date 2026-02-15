# Graph Sentinel Architecture

Graph Sentinel is a production-grade mule account detection system that uses graph-based analysis, machine learning, and behavioral modeling to identify suspicious financial accounts involved in money laundering networks.

## Problem Statement

Mule accounts are bank accounts used to launder money, typically characterized by:
- High inbound volume from many sources with rapid outflows to few beneficiaries (funnel patterns)
- Circular money flows between interconnected accounts
- Multi-hop layering chains where amounts decay at each step
- Dormant accounts suddenly activated for illicit transfers
- Multiple accounts sharing device fingerprints or IP addresses

Traditional rule-based systems fail to capture the structural and temporal complexity of these patterns operating across a network. Graph Sentinel addresses this by modeling transactions as a directed weighted graph and applying network analysis, propagation algorithms, and ensemble machine learning.

## System Overview

```
                                    ┌─────────────────────────────────────────────┐
                                    │              Data Layer                     │
                                    │  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │
                                    │  │PostgreSQL│  │  Redis  │  │   Neo4j     │  │
                                    │  │(storage) │  │ (cache) │  │(graph store)│  │
                                    │  └─────────┘  └─────────┘  └─────────────┘  │
                                    └─────────────────────────────────────────────┘
                                                          │
┌──────────────────┐                                      │
│   Simulation     │─────────────┐                        │
│   (synthetic     │             │                        │
│    data gen)     │             │                        │
└──────────────────┘             ▼                        ▼
                          ┌─────────────┐          ┌─────────────┐
                          │  Ingestion  │          │    Graph    │
                          │  Pipeline   │─────────▶│   Builder   │
                          └─────────────┘          └─────────────┘
                                                         │
                    ┌────────────────────────────────────┼────────────────────────────────────┐
                    │                                    │                                    │
                    ▼                                    ▼                                    ▼
           ┌────────────────┐                  ┌────────────────┐                  ┌────────────────┐
           │    Feature     │                  │     Graph      │                  │    Sequence    │
           │   Engineering  │                  │   Embeddings   │                  │    Encoder     │
           │  (36+ features)│                  │  (64-dim vecs) │                  │ (32-dim vecs)  │
           └────────────────┘                  └────────────────┘                  └────────────────┘
                    │                                    │                                    │
                    │                    ┌──────────────┼──────────────┐                     │
                    │                    │              │              │                     │
                    │                    ▼              ▼              ▼                     │
                    │           ┌───────────────┐ ┌──────────┐ ┌──────────────┐              │
                    │           │   Supervised  │ │ Graph    │ │ Unsupervised │              │
                    │           │   Training    │ │Classifier│ │  Anomaly     │              │
                    │           └───────────────┘ └──────────┘ └──────────────┘              │
                    │                    │              │              │                     │
                    │                    └──────────────┼──────────────┘                     │
                    │                                   │                                    │
                    ▼                                   ▼                                    ▼
           ┌────────────────┐                  ┌────────────────┐                  ┌────────────────┐
           │  Propagation   │                  │    Scoring     │                  │   Sequence     │
           │   Engine       │─────────────────▶│ (Meta-model +  │◀─────────────────│   Anomaly      │
           │(belief diffusion)                 │  Calibration)  │                  │   Scores       │
           └────────────────┘                  └────────────────┘                  └────────────────┘
                                                       │
                                                       ▼
                                              ┌────────────────┐
                                              │     Alerts     │
                                              │   + API Layer  │
                                              └────────────────┘
```

## Module Breakdown

### 1. Simulation (`src/simulation/`)

Generates realistic synthetic financial data for training and evaluation when real data is unavailable.

#### Components

| File | Purpose |
|------|---------|
| `orchestrator.py` | Coordinates account generation, transaction simulation, and fraud injection |
| `accounts.py` | Generates accounts with behavioral profiles (income level, activity patterns, KYC tier) |
| `transactions.py` | Simulates normal transactions following power-law and temporal distributions |
| `behavioral.py` | Models account behavioral profiles (salary cycles, spending patterns) |
| `validator.py` | Validates generated data for statistical properties |

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_accounts` | 50,000 | Number of synthetic accounts |
| `simulation_days` | 180 | Time span of simulation |
| `total_transactions` | 2,000,000 | Target transaction count |
| `fraud_ratio` | 0.015 | Percentage of accounts flagged as mules |

### 2. Fraud Pattern Injection (`src/fraud/`)

Injects structured fraud patterns into the transaction network to create labeled training data.

#### Pattern Types

| Pattern | File | Description |
|---------|------|-------------|
| Funnel | `patterns/funnel.py` | High inbound from many sources, rapid outbound to few beneficiaries |
| Circular Flow | `patterns/circular.py` | Strongly connected components with 3-8 hop cycles |
| Layering | `patterns/layering.py` | Multi-hop chains (3-7 hops) with decaying amounts and high velocity |
| Dormant Activation | `patterns/dormant.py` | Accounts inactive for 60+ days suddenly activated |
| Device Sharing | `patterns/device_sharing.py` | Multiple accounts using same device fingerprint |

Each pattern assigns accounts to specific roles (mule, beneficiary, originator) and generates transactions matching the pattern's signature.

### 3. Graph Construction (`src/graph/`)

Converts raw transactions into a directed weighted graph representation.

#### `GraphBuilder`

Builds a NetworkX DiGraph where:
- **Nodes** represent accounts with attributes: `account_type`, `kyc_level`, `region`, `is_fraud`, `fraud_role`
- **Edges** represent aggregated transaction flows with attributes:
  - `weight`: Total transaction volume
  - `frequency`: Number of transactions
  - `avg_amount`, `min_amount`, `max_amount`: Amount statistics
  - `first_txn`, `last_txn`: Temporal bounds
  - `avg_inter_txn_gap`: Average time between transactions

Supports incremental graph updates for streaming scenarios.

#### `TemporalGraph`

Builds rolling-window graph snapshots (7, 30, 90 days) to capture temporal patterns.

### 4. Feature Engineering (`src/features/`)

Extracts 36+ features from the graph and transaction data across three categories.

#### Structural Features (`structural.py`)

| Feature | Computation |
|---------|-------------|
| `in_degree`, `out_degree`, `total_degree` | Node degree counts |
| `weighted_in_degree`, `weighted_out_degree` | Volume-weighted degrees |
| `degree_ratio` | out_degree / in_degree |
| `betweenness` | Betweenness centrality (sampled for large graphs) |
| `pagerank` | PageRank score |
| `clustering_coefficient` | Local clustering on undirected projection |
| `k_core_number` | K-core decomposition |
| `scc_size` | Strongly connected component size |
| `community_id` | Louvain community assignment |
| `subgraph_density` | Edge density within community |

#### Behavioral Features (`behavioral.py`)

| Feature | Description |
|---------|-------------|
| `incoming_volume`, `outgoing_volume` | Total value in/out |
| `avg_incoming_amount`, `avg_outgoing_amount` | Mean transaction value |
| `max_incoming_amount`, `max_outgoing_amount` | Largest single transaction |
| `incoming_txn_count`, `outgoing_txn_count` | Transaction frequency |
| `incoming_counterparty_diversity`, `outgoing_counterparty_diversity` | Unique counterparties |
| `net_balance_ratio` | (in - out) / (in + out) |
| `velocity_score` | out / in ratio |
| `receive_to_send_delay_hours` | Time between receiving and forwarding funds |
| `burst_index` | Transaction interval variability |
| `dormancy_score` | Maximum gap relative to activity span |
| `repeat_transfer_ratio` | Fraction of transfers to repeat counterparties |

#### Fraud-Specific Features (`fraud_specific.py`)

| Feature | Description |
|---------|-------------|
| `circularity_score` | Participation in detected cycles (up to 8 hops) |
| `funnel_score` | Concentration metric: high in-degree, low out-degree, balanced flow |
| `layering_depth` | Maximum directed path length passing through node |
| `risk_propagation_score` | Basic diffusion from flagged neighbors |
| `device_cluster_score` | Number of accounts sharing device fingerprints |
| `suspicious_chain_length` | Length of longest suspicious transaction chain |

### 5. Graph Learning (`src/graph_learning/`)

Generates node embeddings that capture structural position in the network.

#### `GraphEmbeddingEngine` (`embeddings.py`)

Produces 64-dimensional node embeddings by combining:

1. **Spectral Embedding (32 dims)**: Eigenvectors of the normalized graph Laplacian capturing global structure
2. **Random Walk Embedding (32 dims)**: TruncatedSVD on co-occurrence matrix from random walks (10 walks per node, 40 steps each)

#### `GraphClassifier` (`model.py`)

XGBoost classifier that operates on concatenated embeddings and node features:
- 200 estimators, max depth 6, learning rate 0.05
- Handles class imbalance via `scale_pos_weight`
- 3-fold stratified cross-validation for evaluation
- Outputs fraud probability per node

### 6. Sequence Modeling (`src/sequence/`)

Captures temporal behavioral patterns from transaction sequences.

#### `SequenceBuilder` (`builder.py`)

Converts transaction history into fixed-length (200) numerical sequences per account, with 5 features per transaction:
- `amount`: Log-scaled, normalized amount
- `is_sender`: Direction indicator (1 = sent, 0 = received)
- `hour_of_day`: Normalized hour (0-23 → 0-1)
- `day_of_week`: Normalized weekday (0-6 → 0-1)
- `time_delta_hours`: Normalized inter-transaction gap

#### `SequenceEncoder` (`encoder.py`)

TruncatedSVD-based autoencoder that:
1. Computes statistical summaries per feature channel (mean, std, min, max, skew, slope)
2. Reduces to 32-dimensional behavioral embedding
3. Computes anomaly scores via reconstruction error

### 7. Risk Propagation (`src/propagation/`)

#### `RiskPropagationEngine`

Iterative belief propagation algorithm that diffuses risk from seeded fraud nodes:

1. **Initialization**: Known fraud nodes seeded with score 1.0
2. **Transition Matrix**: Column-normalized edge weights with log-scaled strength
3. **Propagation**: Iterative update with decay factor (default 0.5)
4. **Hop Decay**: Additional attenuation based on shortest-path distance from seeds
5. **Convergence**: Stops when score delta < 1e-5 or after 20 iterations

Key parameters:
- `decay_factor`: 0.5 (risk halves per hop)
- `max_hops`: 5
- `max_iterations`: 20

### 8. Model Training (`src/training/`)

#### Supervised Training (`supervised.py`)

Trains three classifiers on extracted features:

| Model | Configuration |
|-------|---------------|
| **XGBoost** | 300 estimators, depth 6, LR 0.05, early stopping on AUCPR |
| **LightGBM** | 300 estimators, depth 6, LR 0.05, average precision metric |
| **Random Forest** | 300 estimators, balanced class weights |

All models use sample weighting for class imbalance. Best model selected by PR-AUC.

#### Unsupervised Detection (`unsupervised.py`)

Ensemble anomaly detection:

| Model | Weight | Configuration |
|-------|--------|---------------|
| **Isolation Forest** | 0.4 | 200 estimators, 2% contamination |
| **Local Outlier Factor** | 0.3 | 20 neighbors, 2% contamination |
| **DBSCAN Density Deviation** | 0.3 | eps=2.0, min_samples=5 |

Combined score normalized to [0, 1].

### 9. Scoring Engine (`src/scoring/`)

#### `RiskScoringEngine` (`risk_engine.py`)

Aggregates all signals into final risk scores (0-100 scale):

**Signal Inputs:**
- Behavioral component score
- Structural component score
- Fraud-specific/propagation score
- Graph classifier probabilities
- Sequence anomaly scores
- Supervised model probabilities
- Unsupervised anomaly scores

**Scoring Modes:**

1. **Static Weights** (fallback): Weighted average with configurable weights (default: behavioral 0.35, structural 0.35, propagation 0.30)
2. **Meta-Model** (when trained): Learned signal blending via LightGBM

#### `MetaScorer` (`meta_model.py`)

LightGBM stacking model that learns optimal signal combination:
- 200 estimators, depth 4, learning rate 0.03
- 5-fold stratified CV for training
- Outputs raw fraud probabilities

#### `ScoreCalibrator` (`calibration.py`)

Isotonic regression calibration ensuring score reliability:
- Maps raw probabilities to calibrated probabilities
- Computes Expected Calibration Error (ECE) and Brier score
- Ensures score of 0.8 corresponds to ~80% observed fraud rate

### 10. Monitoring (`src/monitoring/`)

#### Feature Drift Detection (`drift.py`)

Monitors feature distribution changes between baseline and current data:

| Metric | Purpose |
|--------|---------|
| **KS Statistic** | Distribution divergence test |
| **PSI (Population Stability Index)** | Distribution shift magnitude |
| **Wasserstein Distance** | Earth-mover distance between distributions |
| **Relative Mean Shift** | Normalized mean change |

Drift threshold: KS p-value < 0.05, relative shift > 2.0, or PSI > 0.2.

#### Graph Drift Detection (`graph_drift.py`)

Monitors graph structural changes:
- Node/edge counts, density
- Degree distribution (mean, max, std)
- Clustering coefficient
- Component metrics

Flags drift when 3+ metrics show >30% relative change.

#### Prediction Drift (`prediction_drift.py`)

Monitors score distribution stability over time.

#### Alert Service (`alerts.py`)

Generates alerts for accounts exceeding threshold:
- Alert threshold: 75.0 (default)
- High-risk threshold: 90.0
- Alert types: `network_risk`, `structural_anomaly`, `behavioral_anomaly`
- Prioritization: Top 100 daily alerts by score

### 11. Explainability (`src/explainability/`)

#### `RiskDecomposer` (`decomposition.py`)

Breaks down risk scores into contributing factors:
- Component scores (behavioral, structural, propagation)
- Top 10 contributing features by absolute magnitude
- SHAP values (when model available)

#### Report Generation (`reports.py`)

Generates markdown reports with score decomposition details.

### 12. API Layer (`src/api/`)

FastAPI REST service for production deployment.

#### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/health/ready` | GET | Readiness check |
| `/api/v1/scores/{account_id}` | GET | Single account risk score |
| `/api/v1/scores/batch` | POST | Batch risk scores |
| `/api/v1/scores` | GET | Top risk scores (paginated) |
| `/api/v1/scores/{account_id}/decomposition` | GET | Score decomposition |
| `/api/v1/alerts` | GET | Active alerts |
| `/api/v1/alerts/stats` | GET | Alert statistics |
| `/api/v1/simulation/run` | POST | Trigger simulation |
| `/api/v1/simulation/status/{job_id}` | GET | Simulation status |

### 13. Storage Layer (`src/storage/`)

| Component | Purpose |
|-----------|---------|
| `postgres.py` | SQLAlchemy async connection for persistent storage |
| `redis_cache.py` | Redis caching for computed scores and features |

### 14. Configuration (`src/config/settings.py`)

Pydantic-based configuration with environment variable support (prefix: `GS_`).

#### Configuration Groups

| Group | Purpose |
|-------|---------|
| `SimulationConfig` | Data generation parameters |
| `FraudConfig` | Fraud pattern ratios and parameters |
| `GraphConfig` | Rolling window sizes, edge filters |
| `GraphLearningConfig` | Embedding dimensions, walk parameters |
| `SequenceConfig` | Sequence length, embedding dimensions |
| `PropagationConfig` | Decay factors, iteration limits |
| `ScoringConfig` | Weights, thresholds, model toggles |
| `Neo4jConfig` | Graph database connection |
| `PostgresConfig` | Relational database connection |
| `RedisConfig` | Cache connection |

## Data Flow

### Training Pipeline

1. **Simulation** → Generate accounts and transactions with injected fraud patterns
2. **Graph Building** → Construct directed weighted graph from transactions
3. **Feature Engineering** → Extract structural, behavioral, fraud-specific features
4. **Graph Embeddings** → Compute spectral + random walk embeddings
5. **Sequence Encoding** → Build behavioral sequences and encode
6. **Risk Propagation** → Diffuse risk from known fraud nodes
7. **Model Training** → Train supervised classifiers, unsupervised detectors, graph classifier
8. **Meta-Model** → Learn optimal signal blending
9. **Calibration** → Fit isotonic regression on validation set
10. **Evaluation** → Compute PR-AUC, ROC-AUC, precision/recall by threshold

### Inference Pipeline

1. **Ingestion** → Load transaction and account data
2. **Graph Update** → Build or incrementally update graph
3. **Feature Extraction** → Compute all features
4. **Embedding Inference** → Generate embeddings for new/updated nodes
5. **Model Inference** → Score through all models
6. **Signal Aggregation** → Meta-model combines signals
7. **Calibration** → Apply isotonic calibration
8. **Alert Generation** → Flag accounts above threshold
9. **API Serving** → Expose scores and alerts

## Machine Learning Models Summary

| Model | Type | Library | Purpose |
|-------|------|---------|---------|
| XGBoost Classifier | Supervised | xgboost | Primary fraud classifier |
| LightGBM Classifier | Supervised | lightgbm | Alternative classifier |
| Random Forest | Supervised | scikit-learn | Ensemble diversity |
| Graph Classifier | Semi-supervised | xgboost | Embedding-based node classification |
| Meta-Scorer | Stacking | lightgbm | Signal combination |
| Isotonic Regression | Calibration | scikit-learn | Probability calibration |
| Isolation Forest | Unsupervised | scikit-learn | Global anomaly detection |
| Local Outlier Factor | Unsupervised | scikit-learn | Local density anomalies |
| DBSCAN | Unsupervised | scikit-learn | Cluster-based outliers |
| TruncatedSVD | Dimensionality reduction | scikit-learn | Sequence encoding, random walk embeddings |

## Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Data Processing | Polars, NumPy, Pandas |
| Graph Analysis | NetworkX |
| ML Framework | scikit-learn, XGBoost, LightGBM |
| API Framework | FastAPI, Uvicorn |
| Explainability | SHAP |
| Visualization | Plotly, PyVis, Matplotlib |
| Graph Database | Neo4j |
| Relational Database | PostgreSQL (asyncpg) |
| Cache | Redis |
| Configuration | Pydantic, pydantic-settings |
| Logging | structlog |
| CLI | Click |
| Testing | pytest, pytest-asyncio |
| Code Quality | Ruff, mypy, pre-commit |

## Directory Structure

```
src/
├── api/                  # REST API (FastAPI)
│   ├── routes/           # Endpoint handlers
│   └── schemas.py        # Request/response models
├── config/               # Configuration management
├── explainability/       # SHAP decomposition, reports
├── features/             # Feature extraction
│   ├── structural.py     # Graph-based features
│   ├── behavioral.py     # Transaction-based features
│   ├── fraud_specific.py # Pattern-specific features
│   └── pipeline.py       # Feature aggregation
├── fraud/                # Fraud pattern injection
│   └── patterns/         # Individual pattern implementations
├── graph/                # Graph construction
├── graph_learning/       # Embeddings and graph classification
├── ingestion/            # Data loading
├── monitoring/           # Drift detection, alerts
├── propagation/          # Risk diffusion algorithms
├── scoring/              # Score computation and calibration
├── sequence/             # Behavioral sequence modeling
├── simulation/           # Synthetic data generation
├── storage/              # Database interfaces
├── training/             # Model training pipelines
├── visualization/        # Plotting utilities
└── cli.py                # Command-line interface
```

## Performance Characteristics

- **Graph building**: O(E) where E is edge count
- **Feature extraction**: O(N + E) for most features, O(N * log N) for centrality measures
- **Propagation**: O(iterations * N * avg_degree)
- **Embedding computation**: O(N * walks * walk_length) for random walks, O(N^2) worst case for spectral
- **Model inference**: O(N * features * trees) for gradient boosted models

For 50,000 accounts and 2,000,000 transactions, full pipeline execution takes approximately 2-5 minutes on standard hardware.

## Validation and Testing

The system includes comprehensive validation:
- Statistical validation of generated data (power-law degree distribution, temporal patterns)
- Unit tests for all major components
- Integration tests for pipeline execution
- Drift detection for production monitoring

Test coverage spans simulation, graph construction, feature engineering, scoring, and API routes.
