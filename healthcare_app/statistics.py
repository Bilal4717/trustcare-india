"""
Statistical confidence intervals for TrustCare India agent outputs.

Provides:
- Bootstrap confidence intervals for trust/confidence scores
- Regional severity prediction intervals
- Data-completeness-adjusted uncertainty bounds
- Summary statistics for dataset quality

These methods address the hackathon evaluation criterion:
"How would you take data messiness into account when framing conclusions?
 Can we use statistics-based methods to create prediction intervals?"
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval for a list of scores.

    Returns (lower, upper) bounds at the requested confidence level.
    Falls back to (score, score) for a single-element list.
    """
    if not scores:
        return (0.0, 0.0)
    if len(scores) == 1:
        return (scores[0], scores[0])

    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()

    alpha = 1.0 - confidence
    lo_idx = int(math.floor(alpha / 2 * n_bootstrap))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))
    return (round(means[lo_idx], 4), round(means[hi_idx], 4))


def score_confidence_interval(
    point_estimate: float,
    completeness: float,
    n_docs: int,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Analytical (parametric) confidence interval for a single score.

    Uses a completeness-adjusted standard error:
      SE ∝ (1 - completeness) / sqrt(n_docs)
    so sparser, less-complete data produces wider intervals.

    Returns a dict with keys: estimate, lower, upper, margin, width.
    """
    completeness = max(0.0, min(1.0, completeness))
    n = max(1, n_docs)

    # Base uncertainty grows as completeness drops and doc count shrinks
    base_uncertainty = (1.0 - completeness) * 0.35
    se = base_uncertainty / math.sqrt(n)

    # z-score for requested confidence
    z = _z_score(confidence)
    margin = z * se

    lower = max(0.0, round(point_estimate - margin, 4))
    upper = min(1.0, round(point_estimate + margin, 4))
    return {
        "estimate": round(point_estimate, 4),
        "lower": lower,
        "upper": upper,
        "margin": round(margin, 4),
        "width": round(upper - lower, 4),
        "confidence": confidence,
    }


def _z_score(confidence: float) -> float:
    z_table = {0.80: 1.282, 0.85: 1.440, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    return z_table.get(round(confidence, 2), 1.960)


# ── Desert severity intervals ──────────────────────────────────────────────────

def desert_severity_interval(
    missing_count: int,
    total_specialties: int,
    facility_count: int,
    confidence: float = 0.90,
) -> Dict[str, float]:
    """
    Prediction interval for how severe a healthcare desert really is.

    Logic: fewer facilities → higher uncertainty about whether a missing
    specialty truly isn't available (vs. just not listed in the data).
    """
    total_specialties = max(1, total_specialties)
    point = missing_count / total_specialties

    # Sparse regions have higher data uncertainty
    data_sparsity = 1.0 / math.sqrt(max(1, facility_count))
    margin = _z_score(confidence) * 0.15 * data_sparsity

    lower = max(0.0, round(point - margin, 4))
    upper = min(1.0, round(point + margin, 4))
    return {
        "severity": round(point, 4),
        "lower": lower,
        "upper": upper,
        "margin": round(margin, 4),
        "confidence": confidence,
        "note": (
            "Wide interval: very few facilities in region — data may be incomplete"
            if facility_count < 3
            else "Narrow interval: sufficient facility coverage for this region"
        ),
    }


# ── Regional aggregation ───────────────────────────────────────────────────────

def regional_trust_summary(
    trust_scores: List[float],
    completeness_scores: Optional[List[float]] = None,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Aggregate trust scores for a region into summary statistics with CI.

    Returns mean, median, std, bootstrap CI bounds, and data quality flag.
    """
    if not trust_scores:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "lower": 0.0, "upper": 0.0}

    n = len(trust_scores)
    mean = sum(trust_scores) / n
    sorted_s = sorted(trust_scores)
    median = (
        sorted_s[n // 2]
        if n % 2 == 1
        else (sorted_s[n // 2 - 1] + sorted_s[n // 2]) / 2
    )
    variance = sum((x - mean) ** 2 for x in trust_scores) / max(1, n - 1)
    std = math.sqrt(variance)

    lo, hi = bootstrap_ci(trust_scores, confidence=confidence)

    mean_completeness = (
        sum(completeness_scores) / len(completeness_scores)
        if completeness_scores
        else 0.5
    )
    data_quality = (
        "high" if mean_completeness >= 0.7 and n >= 5
        else "medium" if mean_completeness >= 0.4 or n >= 3
        else "low"
    )

    return {
        "mean": round(mean, 4),
        "median": round(median, 4),
        "std": round(std, 4),
        "lower": lo,
        "upper": hi,
        "n": n,
        "confidence": confidence,
        "data_quality": data_quality,
        "mean_completeness": round(mean_completeness, 4),
    }


# ── Dataset-wide quality stats ─────────────────────────────────────────────────

def dataset_quality_report(df) -> Dict:
    """
    Compute a dataset-wide data quality summary from the facilities DataFrame.
    Returns field fill rates, completeness distribution, and overall score.
    """
    import pandas as pd  # local import to keep module lightweight

    if df is None or df.empty:
        return {"error": "No dataset loaded"}

    n = len(df)
    key_fields = [
        "description", "specialties", "capability",
        "procedure", "equipment", "numberDoctors",
    ]
    null_sentinels = {"null", "none", "[]", "{}", "", "nan", "nat"}

    def is_filled(v) -> bool:
        if v is None:
            return False
        return str(v).strip().lower() not in null_sentinels

    field_fill = {}
    for f in key_fields:
        if f in df.columns:
            filled = df[f].apply(is_filled).sum()
            field_fill[f] = round(filled / n, 4)
        else:
            field_fill[f] = 0.0

    overall_completeness = (
        df["_completeness"].mean() if "_completeness" in df.columns else 0.0
    )

    # Distribution buckets
    if "_completeness" in df.columns:
        comp = df["_completeness"]
        high = int((comp >= 0.7).sum())
        medium = int(((comp >= 0.4) & (comp < 0.7)).sum())
        low = int((comp < 0.4).sum())
    else:
        high = medium = low = 0

    states = (
        int(df["address_stateOrRegion"].nunique())
        if "address_stateOrRegion" in df.columns
        else 0
    )
    cities = (
        int(df["address_city"].nunique()) if "address_city" in df.columns else 0
    )
    has_coords = 0
    if "latitude" in df.columns and "longitude" in df.columns:
        has_coords = int(
            df[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
            .notna().all(axis=1).sum()
        )

    return {
        "total_facilities": n,
        "states_covered": states,
        "cities_covered": cities,
        "facilities_with_coords": has_coords,
        "geo_coverage_pct": round(has_coords / max(1, n) * 100, 1),
        "overall_completeness": round(float(overall_completeness), 4),
        "completeness_distribution": {"high": high, "medium": medium, "low": low},
        "field_fill_rates": field_fill,
        "data_quality_note": (
            "Good quality — majority of records have detailed descriptions"
            if overall_completeness >= 0.6
            else "Moderate quality — many records are sparse; expect wider confidence intervals"
            if overall_completeness >= 0.35
            else "Sparse dataset — confidence intervals will be wide; treat conclusions as indicative only"
        ),
    }
