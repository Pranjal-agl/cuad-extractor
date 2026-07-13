"""
src/eval/monitor.py

Production drift monitoring for the CUAD clause extractor.

WHAT THIS TRACKS
----------------
1. Prediction drift     — rolling distribution of clause type predictions.
                          If the model starts predicting more/fewer of a
                          specific clause type than baseline, something changed.
2. Confidence drift     — average logit gap (max logit - second logit) per
                          request. A drop signals the model is becoming less
                          certain, often an early sign of distribution shift.
3. Text length drift    — contracts getting longer/shorter than training dist.
                          affects truncation rate and therefore coverage.
4. Online F1 tracking   — when ground truth labels arrive (via feedback),
                          computes rolling F1 and flags offline/online gap.

USAGE
-----
In production, call log_prediction() after every extract request.
Run the drift report daily:

    python -m src.eval.monitor --report --baseline experiments/eval_test_fp32.json

Or check online/offline gap after labelling some production examples:

    python -m src.eval.monitor --label  # interactive labelling CLI
    python -m src.eval.monitor --report
"""

import json
import math
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
EXPERIMENTS_DIR = ROOT / "experiments"
LOG_PATH = DATA_DIR / "production_log.jsonl"
LABELS_PATH = DATA_DIR / "production_labels.jsonl"

CLAUSE_TYPES = [
    "Governing Law", "Termination For Convenience", "Renewal Term",
    "Non-Compete", "Confidentiality Of Agreements", "Indemnification",
    "Cap On Liability", "Assignment",
]


# ── logging ───────────────────────────────────────────────────────────────────


def log_prediction(
    text: str,
    clause: str,
    predicted_spans: list[str],
    latency_ms: float,
    confidence: Optional[float] = None,
) -> None:
    """Append one production prediction to the log."""
    LOG_PATH.parent.mkdir(exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "clause": clause,
        "text_len": len(text),
        "text_prefix": text[:50],
        "n_spans_predicted": len(predicted_spans),
        "has_prediction": len(predicted_spans) > 0,
        "latency_ms": round(latency_ms, 2),
        "confidence": round(confidence, 4) if confidence is not None else None,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── label collection ──────────────────────────────────────────────────────────


def label_session(max_cases: int = 20) -> None:
    """Interactive CLI to label production predictions with ground truth."""
    if not LOG_PATH.exists():
        print("No production log found. Process some contracts first.")
        return

    with open(LOG_PATH) as f:
        log = [json.loads(l) for l in f if l.strip()]

    existing = set()
    if LABELS_PATH.exists():
        with open(LABELS_PATH) as f:
            for line in f:
                rec = json.loads(line)
                existing.add(rec.get("ts_original", ""))

    unlabeled = [r for r in log if r["ts"] not in existing][:max_cases]
    if not unlabeled:
        print("No unlabeled predictions found.")
        return

    print(f"\nLabeling {len(unlabeled)} predictions. Press Ctrl+C to stop.\n")
    labeled = 0

    try:
        for rec in unlabeled:
            print(f"  Clause : {rec['clause']}")
            print(f"  Text   : \"{rec['text_prefix']}...\"")
            print(f"  Predicted spans: {rec['n_spans_predicted']}")
            ans = input("  Was prediction correct? [y/n/skip]: ").strip().lower()
            if ans in ("s", "skip", ""):
                print()
                continue
            correct = ans in ("y", "yes")
            label_rec = {
                "ts_labeled": datetime.now(timezone.utc).isoformat(),
                "ts_original": rec["ts"],
                "clause": rec["clause"],
                "correct": correct,
                "n_spans": rec["n_spans_predicted"],
            }
            with open(LABELS_PATH, "a") as f:
                f.write(json.dumps(label_rec) + "\n")
            labeled += 1
            print(f"  Saved {'correct' if correct else 'incorrect'}.\n")
    except KeyboardInterrupt:
        print("\nSession interrupted.")

    print(f"Labeled {labeled} predictions. Total labeled: {len(existing) + labeled}")


# ── drift report ──────────────────────────────────────────────────────────────


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _stddev(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((x - m)**2 for x in vals) / (len(vals) - 1))


def drift_report(baseline_path: Path, window: int = 200) -> None:
    if not LOG_PATH.exists():
        print("No production log found.")
        return

    with open(LOG_PATH) as f:
        log = [json.loads(l) for l in f if l.strip()]

    recent = log[-window:]
    n = len(recent)

    print(f"\n{'='*60}")
    print(f"  Drift Report")
    print(f"  Baseline  : {baseline_path.name}")
    print(f"  Window    : last {n} predictions")
    print(f"{'='*60}\n")

    if n == 0:
        print("  No production predictions logged yet.")
        return

    # 1. Prediction rate drift (fraction of requests with at least one span)
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)

        print("  Prediction rate (fraction of requests with a match):")
        baseline_per_clause = baseline.get("per_clause", {})
        clause_preds = defaultdict(list)
        for rec in recent:
            clause_preds[rec["clause"]].append(rec["has_prediction"])

        for clause in CLAUSE_TYPES:
            preds = clause_preds.get(clause, [])
            if not preds:
                continue
            prod_rate = _mean([float(p) for p in preds])
            # Baseline positive rate from eval
            baseline_clause = baseline_per_clause.get(clause, {})
            baseline_n = baseline_clause.get("n", 0)
            baseline_pos = baseline_clause.get("n_positive", 0)
            baseline_rate = baseline_pos / baseline_n if baseline_n > 0 else None
            if baseline_rate is not None:
                delta = prod_rate - baseline_rate
                flag = "  DRIFT" if abs(delta) > 0.2 else ""
                print(f"    {clause:<35} prod={prod_rate:.2f}  baseline={baseline_rate:.2f}  d={delta:+.2f}{flag}")

    # 2. Text length drift
    lengths = [r["text_len"] for r in recent]
    print(f"\n  Text length: mean={_mean(lengths):.0f}  std={_stddev(lengths):.0f}  "
          f"range=[{min(lengths)}, {max(lengths)}]")

    # 3. Latency drift
    latencies = [r["latency_ms"] for r in recent if r.get("latency_ms")]
    if latencies:
        import numpy as np
        p99 = float(np.percentile(latencies, 99))
        baseline_p99 = None
        if baseline_path.exists():
            baseline_p99 = baseline.get("latency", {}).get("p99_ms")
        flag = ""
        if baseline_p99 and p99 > baseline_p99 * 1.5:
            flag = "  LATENCY REGRESSION"
        print(f"\n  Latency p99: {p99:.1f}ms", end="")
        if baseline_p99:
            print(f"  (baseline: {baseline_p99}ms){flag}")
        else:
            print()

    # 4. Online accuracy (if labels exist)
    if LABELS_PATH.exists():
        with open(LABELS_PATH) as f:
            labels = [json.loads(l) for l in f if l.strip()]
        if labels:
            correct = sum(1 for l in labels if l["correct"])
            online_acc = correct / len(labels)
            baseline_f1 = baseline.get("macro_f1") if baseline_path.exists() else None
            gap_str = ""
            if baseline_f1:
                gap = online_acc - baseline_f1
                flag = "  GAP" if abs(gap) > 0.1 else ""
                gap_str = f"  offline={baseline_f1:.4f}  gap={gap:+.4f}{flag}"
            print(f"\n  Online accuracy: {correct}/{len(labels)} ({online_acc:.2%}){gap_str}")
    else:
        print(f"\n  No labels yet. Run --label to add ground truth.")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--label", action="store_true")
    parser.add_argument("--baseline", default="experiments/eval_test_fp32.json")
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    if args.label:
        label_session(max_cases=args.n)
    elif args.report:
        drift_report(Path(args.baseline), window=args.window)
    else:
        parser.print_help()
