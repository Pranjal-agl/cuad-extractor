"""
src/eval/calibration.py

Not applicable in the same sense as a binary fraud classifier since this is
extractive span extraction, not a single score. This module instead reports
per-clause F1 as a calibration proxy: how well precision and recall track
n_positive frequency in the test set.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"


def calibration_report(eval_path: Path):
    with open(eval_path) as f:
        results = json.load(f)

    print(f"\n{'='*60}")
    print(f"  Calibration Proxy Report")
    print(f"  Source: {eval_path.name}")
    print(f"{'='*60}\n")

    print(f"  {'Clause':<35} {'F1':>7} {'Prec':>7} {'Rec':>7} {'n_pos':>7} {'n':>5}")
    for clause, m in results.get("per_clause", {}).items():
        if m["f1"] is None:
            continue
        print(f"  {clause:<35} {m['f1']:>7.3f} {m['precision']:>7.3f} {m['recall']:>7.3f} "
              f"{m.get('n_positive', 0):>7} {m['n']:>5}")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default=str(EXPERIMENTS_DIR / "eval_test_fp32.json"))
    parser.add_argument("--plot", action="store_true", help="unused, kept for CLI compatibility")
    args = parser.parse_args()

    path = Path(args.eval_file)
    if not path.exists():
        print(f"Eval file not found: {path}. Run evaluate.py first.")
    else:
        calibration_report(path)
