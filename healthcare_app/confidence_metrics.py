"""
Confidence/uncertainty evaluation helpers.

This module provides lightweight, dataset-driven diagnostics to evaluate
whether confidence intervals are likely calibrated and informative.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List

from healthcare_app.statistics import regional_trust_summary, score_confidence_interval


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _sample_completeness(df, n: int = 150, seed: int = 13) -> List[float]:
    if df is None or len(df) == 0:
        return []
    rng = random.Random(seed)
    series = df["_completeness"].dropna().tolist() if "_completeness" in df.columns else []
    if not series:
        return []
    return [float(rng.choice(series)) for _ in range(min(n, max(20, len(series))))]


def confidence_metrics_report(df, quality_report: Dict) -> Dict:
    """
    Build practical confidence diagnostics from data quality + synthetic replay.

    Notes:
    - Because there is no canonical answer key, coverage is estimated via
      simulation around completeness-conditioned pseudo-point-estimates.
    - This is intended for operational monitoring and rubric transparency.
    """
    samples = _sample_completeness(df)
    if not samples:
        return {
            "status": "insufficient_data",
            "message": "No completeness samples available to estimate confidence diagnostics.",
        }

    rng = random.Random(97)
    intervals = []
    covered = 0
    widths = []
    point_scores = []

    for c in samples:
        # pseudo-point-estimate: higher completeness => likely higher trust confidence
        point = _clamp01(0.25 + 0.65 * c + rng.uniform(-0.08, 0.08))
        n_docs = max(1, int(1 + 10 * c))
        ci = score_confidence_interval(point, c, n_docs, confidence=0.95)
        intervals.append(ci)
        widths.append(float(ci.get("width", 0.0)))
        point_scores.append(point)

        # simulate an observed realization near point; acts as proxy target
        observed = _clamp01(point + rng.gauss(0, 0.07 + (1.0 - c) * 0.08))
        if float(ci["lower"]) <= observed <= float(ci["upper"]):
            covered += 1

    n = len(intervals)
    empirical_coverage = covered / max(1, n)
    avg_width = sum(widths) / max(1, n)
    median_width = sorted(widths)[n // 2]

    # coverage quality labels for non-answer-key settings
    if empirical_coverage >= 0.9:
        calibration = "good"
    elif empirical_coverage >= 0.8:
        calibration = "moderate"
    else:
        calibration = "low"

    overall_completeness = float(quality_report.get("overall_completeness", 0.0) or 0.0)
    retrieval_uncertainty = _clamp01(1.0 - (overall_completeness * 0.8))
    model_variance_uncertainty = _clamp01(0.2 + avg_width * 1.8)
    data_quality_uncertainty = _clamp01(1.0 - overall_completeness)
    overall_uncertainty = _clamp01(
        0.45 * data_quality_uncertainty
        + 0.3 * retrieval_uncertainty
        + 0.25 * model_variance_uncertainty
    )

    regional = regional_trust_summary(point_scores, samples, confidence=0.95)

    return {
        "status": "ok",
        "sample_size": n,
        "target_coverage": 0.95,
        "empirical_coverage_proxy": round(empirical_coverage, 4),
        "calibration_label": calibration,
        "interval_width": {
            "mean": round(avg_width, 4),
            "median": round(median_width, 4),
            "min": round(min(widths), 4),
            "max": round(max(widths), 4),
        },
        "uncertainty_components": {
            "uncertainty_data_quality": round(data_quality_uncertainty, 4),
            "uncertainty_retrieval": round(retrieval_uncertainty, 4),
            "uncertainty_model_variance": round(model_variance_uncertainty, 4),
            "overall_uncertainty_score": round(overall_uncertainty, 4),
        },
        "regional_proxy_summary": regional,
        "notes": [
            "Coverage is a proxy metric due to absence of canonical labels.",
            "Use with benchmark prompt evaluations for full calibration claims.",
        ],
    }

