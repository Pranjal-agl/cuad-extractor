"""
src/metrics.py

Statistics helpers shared by evaluation, quantization and serving.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`.

    This is a confidence interval for the quantity actually reported (mean
    per-example F1). The original code computed a Wilson interval for "the
    fraction of examples with F1 > 0.5" and printed it next to mean F1, which
    is why some reported intervals barely contained their own point estimate
    (e.g. Renewal Term 0.764 with CI [0.744, 0.965]).
    """
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return (0.0, 1.0)
    if len(vals) == 1:
        return (float(vals[0]), float(vals[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_resamples, len(vals)))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    spread = (z * (p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def latency_summary(latencies_ms: Sequence[float], target_ms: float | None = None) -> dict:
    if not latencies_ms:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None, "n": 0}
    arr = np.asarray(latencies_ms, dtype=float)
    out = {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "n": int(len(arr)),
    }
    if target_ms is not None:
        out["target_ms"] = target_ms
        out["target_met"] = out["p99_ms"] <= target_ms
    return out


def expected_calibration_error(
    scores: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> dict:
    """Reliability table + ECE for predicted-span confidence scores.

    `scores` are model confidences in [0, 1] for each predicted span and
    `correct` says whether that span overlapped a gold span of the same
    clause. A well-calibrated model's spans with score ~0.8 are correct ~80%
    of the time, which is what makes a confidence threshold meaningful.
    """
    s = np.asarray(scores, dtype=float)
    c = np.asarray(correct, dtype=float)
    bins = []
    ece = 0.0
    if len(s) == 0:
        return {"ece": None, "n": 0, "bins": []}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (s >= lo) & ((s < hi) if hi < 1.0 else (s <= hi))
        n = int(mask.sum())
        if n == 0:
            continue
        conf, acc = float(s[mask].mean()), float(c[mask].mean())
        ece += (n / len(s)) * abs(conf - acc)
        bins.append({"range": [round(lo, 2), round(hi, 2)], "n": n,
                     "mean_confidence": round(conf, 4), "accuracy": round(acc, 4)})
    return {"ece": round(ece, 4), "n": int(len(s)), "bins": bins}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
