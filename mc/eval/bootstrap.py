"""Paired bootstrap significance testing over queries (10,000 resamples, 95% CI)."""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any


def paired_bootstrap_test(
    scores_a: Dict[str, float],
    scores_b: Dict[str, float],
    num_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Perform paired bootstrap hypothesis test comparing Model A (new) against Model B (baseline).

    Args:
        scores_a: dict mapping query_id -> performance metric (e.g. top-1 IoU or binary hit)
        scores_b: dict mapping query_id -> performance metric
        num_resamples: number of paired bootstrap resamples (default: 10,000)
        confidence_level: confidence level for interval (default: 0.95)
        seed: RNG seed for reproducible sampling

    Returns:
        dict containing:
          - mean_a, mean_b, delta_mean
          - ci_lower, ci_upper (95% CI of delta = mean_a - mean_b)
          - p_value: two-sided empirical bootstrap p-value
          - is_significant: bool (whether 0 is excluded from the CI)
    """
    common_qids = sorted(list(set(scores_a.keys()) & set(scores_b.keys())))
    if not common_qids:
        return {
            "error": "No overlapping query IDs found for paired bootstrap",
            "p_value": 1.0,
            "is_significant": False
        }

    arr_a = np.array([scores_a[qid] for qid in common_qids], dtype=np.float64)
    arr_b = np.array([scores_b[qid] for qid in common_qids], dtype=np.float64)
    n = len(common_qids)

    obs_mean_a = float(np.mean(arr_a))
    obs_mean_b = float(np.mean(arr_b))
    obs_delta = obs_mean_a - obs_mean_b

    rng = np.random.RandomState(seed)
    # Generate bootstrap indices: shape [num_resamples, n]
    # To conserve RAM, chunk bootstrap if num_resamples * n > 10,000,000
    chunk_size = 2000
    bootstrap_deltas = []

    for i in range(0, num_resamples, chunk_size):
        curr_k = min(chunk_size, num_resamples - i)
        boot_indices = rng.randint(0, n, size=(curr_k, n))
        boot_a = np.mean(arr_a[boot_indices], axis=1)
        boot_b = np.mean(arr_b[boot_indices], axis=1)
        bootstrap_deltas.append(boot_a - boot_b)

    deltas = np.concatenate(bootstrap_deltas, axis=0)

    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100
    upper_pct = (1.0 - alpha / 2.0) * 100

    ci_lower = float(np.percentile(deltas, lower_pct))
    ci_upper = float(np.percentile(deltas, upper_pct))

    # Empirical p-value under the null hypothesis (delta == 0)
    # Shift bootstrap distribution to center at 0
    shifted_deltas = deltas - obs_delta
    extreme_count = np.sum(np.abs(shifted_deltas) >= np.abs(obs_delta))
    p_value = float((extreme_count + 1) / (num_resamples + 1))

    is_significant = bool(ci_lower > 0 or ci_upper < 0)

    return {
        "num_queries": n,
        "mean_a": round(obs_mean_a, 4),
        "mean_b": round(obs_mean_b, 4),
        "obs_delta": round(obs_delta, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 5),
        "is_significant": is_significant,
        "confidence_level": confidence_level
    }


def compute_metric_ci(
    scores: Dict[str, float],
    num_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Compute 95% bootstrap confidence interval for a single model's metric."""
    vals = np.array(list(scores.values()), dtype=np.float64)
    if len(vals) == 0:
        return 0.0, 0.0, 0.0

    n = len(vals)
    rng = np.random.RandomState(seed)
    boot_means = []
    chunk_size = 2000

    for i in range(0, num_resamples, chunk_size):
        curr_k = min(chunk_size, num_resamples - i)
        boot_idx = rng.randint(0, n, size=(curr_k, n))
        boot_means.append(np.mean(vals[boot_idx], axis=1))

    all_means = np.concatenate(boot_means, axis=0)
    alpha = 1.0 - confidence_level
    ci_low = float(np.percentile(all_means, (alpha / 2.0) * 100))
    ci_high = float(np.percentile(all_means, (1.0 - alpha / 2.0) * 100))

    return round(float(np.mean(vals)), 4), round(ci_low, 4), round(ci_high, 4)
