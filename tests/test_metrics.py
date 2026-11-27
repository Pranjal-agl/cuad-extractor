import pytest

from src.metrics import bootstrap_ci, expected_calibration_error, latency_summary, wilson_ci


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    vals = [0.2, 0.9, 0.8, 1.0, 0.5, 0.7, 0.95, 0.3]
    lo, hi = bootstrap_ci(vals)
    mean = sum(vals) / len(vals)
    assert lo <= mean <= hi
    assert bootstrap_ci(vals) == (lo, hi)


def test_bootstrap_ci_degenerate():
    assert bootstrap_ci([]) == (0.0, 1.0)
    assert bootstrap_ci([0.4]) == (0.4, 0.4)


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(9, 10)
    assert 0 <= lo < 0.9 < hi <= 1


def test_latency_summary_target():
    s = latency_summary([10.0] * 99 + [200.0], target_ms=150)
    assert s["p50_ms"] == 10.0 and s["n"] == 100
    assert s["target_met"] is (s["p99_ms"] <= 150)
    assert latency_summary([])["p99_ms"] is None


def test_ece_perfect_and_bad():
    assert expected_calibration_error([0.95] * 10, [True] * 10)["ece"] == pytest.approx(0.05)
    assert expected_calibration_error([0.95] * 10, [False] * 10)["ece"] == pytest.approx(0.95)
    assert expected_calibration_error([], [])["ece"] is None
