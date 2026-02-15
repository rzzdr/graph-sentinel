from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class ClusterVisualizer:
    def plot_risk_distribution(
        self,
        risk_scores: pl.DataFrame,
        output_path: Path,
    ) -> Path:
        df = risk_scores.to_pandas()

        fig = px.histogram(
            df,
            x="risk_score",
            color="risk_level",
            nbins=50,
            title="Risk Score Distribution",
            labels={"risk_score": "Risk Score", "count": "Count"},
            color_discrete_map={
                "minimal": "#2ecc71",
                "low": "#f39c12",
                "medium": "#e67e22",
                "high": "#e74c3c",
                "critical": "#8e44ad",
            },
        )
        fig.update_layout(bargap=0.1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info("risk_distribution_plotted", path=str(output_path))
        return output_path

    def plot_fraud_clusters_scatter(
        self,
        features: pl.DataFrame,
        risk_scores: pl.DataFrame,
        output_path: Path,
    ) -> Path:
        merged = features.join(risk_scores, on="account_id", how="left")

        cols_for_pca = [
            c
            for c in features.columns
            if c != "account_id"
            and features[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        ]

        if len(cols_for_pca) < 2:
            logger.warning("insufficient_features_for_scatter")
            return output_path

        X = merged.select(cols_for_pca[:20]).fill_null(0).fill_nan(0).to_numpy()

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)

        df = merged.select("account_id", "risk_score", "risk_level").to_pandas()
        df["pc1"] = coords[:, 0]
        df["pc2"] = coords[:, 1]

        fig = px.scatter(
            df,
            x="pc1",
            y="pc2",
            color="risk_level",
            hover_data=["account_id", "risk_score"],
            title="Fraud Cluster Visualization (PCA)",
            color_discrete_map={
                "minimal": "#2ecc71",
                "low": "#f39c12",
                "medium": "#e67e22",
                "high": "#e74c3c",
                "critical": "#8e44ad",
            },
            opacity=0.6,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info("cluster_scatter_plotted", path=str(output_path))
        return output_path

    def plot_score_components(
        self,
        risk_scores: pl.DataFrame,
        output_path: Path,
        top_n: int = 30,
    ) -> Path:
        top = risk_scores.sort("risk_score", descending=True).head(top_n)
        df = top.to_pandas()

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Behavioral",
                x=df["account_id"].str[:8],
                y=df["behavioral_risk_score"],
            )
        )
        fig.add_trace(
            go.Bar(
                name="Structural",
                x=df["account_id"].str[:8],
                y=df["structural_risk_score"],
            )
        )
        fig.add_trace(
            go.Bar(
                name="Propagation",
                x=df["account_id"].str[:8],
                y=df["network_propagation_score"],
            )
        )

        fig.update_layout(
            barmode="stack",
            title=f"Risk Score Decomposition (Top {top_n})",
            xaxis_title="Account",
            yaxis_title="Score Component",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info("score_components_plotted", path=str(output_path))
        return output_path
