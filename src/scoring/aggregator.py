from __future__ import annotations

import numpy as np


class ScoreAggregator:
    def aggregate(
        self,
        behavioral_scores: list[float],
        structural_scores: list[float],
        propagation_scores: list[float],
        weights: tuple[float, float, float] = (0.35, 0.35, 0.30),
    ) -> list[float]:
        w_b, w_s, w_p = weights
        total_weight = w_b + w_s + w_p

        b = np.array(behavioral_scores)
        s = np.array(structural_scores)
        p = np.array(propagation_scores)

        combined = (w_b * b + w_s * s + w_p * p) / total_weight
        combined = np.clip(combined, 0, 100)

        return [round(float(x), 2) for x in combined]

    def decompose(
        self,
        behavioral_score: float,
        structural_score: float,
        propagation_score: float,
        weights: tuple[float, float, float] = (0.35, 0.35, 0.30),
    ) -> dict[str, float]:
        w_b, w_s, w_p = weights
        total_weight = w_b + w_s + w_p

        return {
            "behavioral_contribution": round(w_b * behavioral_score / total_weight, 2),
            "structural_contribution": round(w_s * structural_score / total_weight, 2),
            "propagation_contribution": round(w_p * propagation_score / total_weight, 2),
            "behavioral_raw": round(behavioral_score, 2),
            "structural_raw": round(structural_score, 2),
            "propagation_raw": round(propagation_score, 2),
        }
