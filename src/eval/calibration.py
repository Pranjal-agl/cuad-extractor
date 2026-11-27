"""
src/eval/calibration.py

Prints how well the model's span confidence scores track reality, from an
eval report written by evaluate.py:

  * reliability table: for predicted spans grouped by confidence, how often
    they actually overlap a gold span of the same clause
  * expected calibration error (ECE)
  * per-clause precision/recall on positives and false-positive rate on
    negatives

Use it to choose a confidence threshold for downstream consumers (the API
returns a score with every span).

    python -m src.eval.calibration --eval-file experiments/eval_test_fp32.json
"""

import argparse
import json
from pathlib import Path

from src.config import EXPERIMENTS_DIR


def calibration_report(eval_path: Path) -> None:
    results = json.loads(eval_path.read_text())
    print(f"\n{'=' * 64}\n  Calibration report -- {eval_path.name}\n{'=' * 64}")

    calib = results.get("calibration")
    if calib and calib.get("bins"):
        print(f"\n  Span confidence reliability (n={calib['n']} predicted spans, ECE={calib['ece']}):")
        print(f"  {'confidence':<12} {'n':>6} {'mean conf':>10} {'precision':>10}")
        for b in calib["bins"]:
            lo, hi = b["range"]
            print(f"  {lo:.1f}-{hi:.1f}      {b['n']:>6} {b['mean_confidence']:>10.3f} {b['accuracy']:>10.3f}")
    else:
        print("\n  No span confidence data in this report (produced by an older evaluate.py).")

    print(f"\n  {'Clause':<28} {'Prec+':>7} {'Rec+':>7} {'F1+':>7} {'FPR-':>7} {'n+':>5} {'n-':>5}")
    for clause, m in results.get("per_clause", {}).items():
        if m.get("f1") is None:
            continue

        def fmt(v):
            return f"{v:>7.3f}" if isinstance(v, (int, float)) else f"{'-':>7}"

        print(f"  {clause:<28} {fmt(m.get('precision_positive', m.get('precision')))}"
              f" {fmt(m.get('recall_positive', m.get('recall')))}"
              f" {fmt(m.get('f1_positive', m.get('f1')))}"
              f" {fmt(m.get('false_positive_rate_negative'))}"
              f" {m.get('n_positive', 0):>5} {m.get('n_negative', 0):>5}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default=str(EXPERIMENTS_DIR / "eval_test_fp32.json"))
    args = parser.parse_args()
    path = Path(args.eval_file)
    if not path.exists():
        raise SystemExit(f"Eval file not found: {path}. Run `python -m src.eval.evaluate` first.")
    calibration_report(path)


if __name__ == "__main__":
    main()
