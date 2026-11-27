"""
src/eval/evaluate.py

Eval harness for the CUAD clause extractor:
  * per-clause span F1 (character overlap) with 95% bootstrap CIs
  * separate scores on positive examples and false-positive rate on negatives
  * calibration of span confidence scores (reliability table + ECE)
  * adversarial cases, including a long document that needs sliding windows
  * val-vs-test gap computed with the SAME metric on both splits

USAGE
-----
    python -m src.eval.evaluate --split test                 # FP32
    python -m src.eval.evaluate --split test --quantize int8 # CPU dynamic INT8
    python -m src.eval.evaluate --split test --quantize nf4  # GPU 4-bit
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from src.config import CLAUSE_TYPES, DATA_DIR, EXPERIMENTS_DIR, P99_LATENCY_TARGET_MS
from src.metrics import bootstrap_ci, expected_calibration_error, latency_summary, wilson_ci
from src.spans import span_f1
from src.training.prepare_data import example_spans


def _gold(example: dict, clause: str) -> list[tuple[int, int]]:
    return [(s["start"], s["end"]) for s in example_spans(example) if s["clause"] == clause]


def run_eval(extractor, examples: list[dict], split_name: str = "test", verbose: bool = True) -> dict:
    """Score `extractor` on examples. Each example asks about one clause type."""
    per_clause: dict[str, list[dict]] = defaultdict(list)
    latencies: list[float] = []
    calib_scores: list[float] = []
    calib_correct: list[bool] = []
    errors: list[dict] = []

    if examples:  # warm-up so one-off first-call costs don't land in p99
        extractor.extract(examples[0]["context"])
    for i, ex in enumerate(examples):
        clause = ex["clause"]
        out = extractor.extract(ex["context"])
        latencies.append(out["latency_ms"])
        pred_spans = [sp for sp in out["spans"] if sp["clause"] == clause]
        gold = _gold(ex, clause)
        scores = span_f1([(sp["start"], sp["end"]) for sp in pred_spans], gold)
        scores["has_answer"] = bool(gold)
        per_clause[clause].append(scores)

        # Calibration over every predicted span of every clause in the window.
        all_gold = defaultdict(list)
        for s in example_spans(ex):
            all_gold[s["clause"]].append((s["start"], s["end"]))
        for sp in out["spans"]:
            if sp.get("score") is None:
                continue
            hit = any(sp["start"] < e and sp["end"] > s for s, e in all_gold.get(sp["clause"], []))
            calib_scores.append(sp["score"])
            calib_correct.append(hit)

        if scores["f1"] < 0.5:
            errors.append({
                "clause": clause, "kind": ex.get("kind", "positive"), "f1": round(scores["f1"], 3),
                "gold": [ex["context"][s:e][:300] for s, e in gold[:2]],
                "pred": [sp["text"][:300] for sp in pred_spans[:2]],
                "context_snippet": ex["context"][:200],
            })
        if verbose and (i + 1) % 50 == 0:
            print(f"  [{split_name}] {i + 1}/{len(examples)}")

    results: dict[str, dict] = {}
    clause_f1s = []
    for clause in CLAUSE_TYPES:
        rows = per_clause.get(clause, [])
        if not rows:
            results[clause] = {"f1": None, "n": 0}
            continue
        f1s = [r["f1"] for r in rows]
        pos = [r for r in rows if r["has_answer"]]
        neg = [r for r in rows if not r["has_answer"]]
        mean_f1 = float(np.mean(f1s))
        clause_f1s.append(mean_f1)
        lo, hi = bootstrap_ci(f1s)
        hits = sum(1 for r in pos if r["f1"] >= 0.5)
        h_lo, h_hi = wilson_ci(hits, len(pos))
        n_fp = sum(1 for r in neg if r["f1"] == 0.0)
        results[clause] = {
            "f1": round(mean_f1, 4),
            "ci_95": [round(lo, 4), round(hi, 4)],
            "n": len(rows),
            "n_positive": len(pos),
            "n_negative": len(neg),
            "f1_positive": round(float(np.mean([r["f1"] for r in pos])), 4) if pos else None,
            "precision_positive": round(float(np.mean([r["precision"] for r in pos])), 4) if pos else None,
            "recall_positive": round(float(np.mean([r["recall"] for r in pos])), 4) if pos else None,
            "hit_rate_positive": round(hits / len(pos), 4) if pos else None,
            "hit_rate_ci_95": [round(h_lo, 4), round(h_hi, 4)] if pos else None,
            "false_positive_rate_negative": round(n_fp / len(neg), 4) if neg else None,
        }

    all_f1 = [r["f1"] for rows in per_clause.values() for r in rows]
    return {
        "split": split_name,
        "macro_f1": round(float(np.mean(clause_f1s)), 4) if clause_f1s else 0.0,
        "micro_f1": round(float(np.mean(all_f1)), 4) if all_f1 else 0.0,
        "n_clause_types_scored": len(clause_f1s),
        "per_clause": results,
        "calibration": expected_calibration_error(calib_scores, calib_correct),
        "latency": latency_summary(latencies, P99_LATENCY_TARGET_MS),
        "n_examples": len(examples),
        "n_errors": len(errors),
        "errors": sorted(errors, key=lambda e: e["f1"])[:20],
    }


# Gold spans follow the CUAD annotation convention: annotators highlight the
# whole clause sentence, not a short phrase. The first version of these cases
# used phrase-level gold ("laws of California"), so a model that correctly
# returned the full sentence was marked as failing.
_FILLER = (
    "The Supplier shall deliver the Products to the delivery address specified in each "
    "Purchase Order. Title and risk of loss shall pass upon delivery. The Customer shall "
    "inspect the Products within ten (10) business days and notify the Supplier in writing "
    "of any non-conforming Products. "
)
_GOV_SENTENCE = "This Agreement shall be governed by the laws of the State of California."


def adversarial_cases() -> list[dict]:
    def case(name, context, clause, answers):
        spans = []
        for a in answers:
            start = context.index(a)
            spans.append({"clause": clause, "start": start, "end": start + len(a)})
        return {"name": name, "context": context, "clause": clause, "spans": spans,
                "answers": answers, "has_answer": bool(answers)}

    multi = ("This Agreement is governed by the laws of the State of Delaware. "
             "Notwithstanding the foregoing, disputes arising outside the United States shall be "
             "governed by the laws of the State of New York.")
    return [
        case("short_single_sentence", _GOV_SENTENCE, "Governing Law", [_GOV_SENTENCE]),
        case("no_clause_present",
             "The parties agree to the terms and conditions set forth herein. "
             "Payment shall be made within 30 days of invoice receipt.", "Governing Law", []),
        case("multiple_spans", multi, "Governing Law", [
            "This Agreement is governed by the laws of the State of Delaware.",
            "Notwithstanding the foregoing, disputes arising outside the United States shall be "
            "governed by the laws of the State of New York."]),
        case("boilerplate_noise",
             "IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the date "
             "first written above. COMPANY A By: Name: Title: Date: COMPANY B By: Name: Title: Date:",
             "Cap On Liability", []),
        case("wrong_clause_query", _GOV_SENTENCE, "Non-Compete", []),
        # ~800 tokens of filler before the clause: invisible to a model that
        # truncates at one window, found by sliding-window inference.
        case("clause_beyond_first_window", _FILLER * 12 + _GOV_SENTENCE, "Governing Law", [_GOV_SENTENCE]),
    ]


def adversarial_eval(extractor, verbose: bool = True) -> dict:
    results = []
    for c in adversarial_cases():
        out = extractor.extract(c["context"])
        pred = [sp for sp in out["spans"] if sp["clause"] == c["clause"]]
        scores = span_f1([(sp["start"], sp["end"]) for sp in pred], _gold(c, c["clause"]))
        passed = scores["f1"] >= 0.5 if c["has_answer"] else not pred
        results.append({
            "case": c["name"], "clause": c["clause"], "expected": c["answers"],
            "predicted": [sp["text"] for sp in pred], "f1": round(scores["f1"], 3),
            "n_windows": out["n_windows"], "latency_ms": round(out["latency_ms"], 2), "pass": passed,
        })
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {c['name']}: f1={scores['f1']:.3f}")
    return {"adversarial_cases": results, "pass_rate": round(sum(r["pass"] for r in results) / len(results), 4)}


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    from src.config import setup_mlflow
    from src.inference import QUANT_MODES, ClauseExtractor
    from src.training.prepare_data import verify_lock

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--quantize", default="none", choices=QUANT_MODES)
    parser.add_argument("--quantized", action="store_true", help="deprecated alias for --quantize nf4")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    quant = "nf4" if args.quantized else args.quantize
    tag = "fp32" if quant == "none" else quant

    verify_lock()
    extractor = ClauseExtractor.load(args.model_path, quantize=quant)
    examples = _load_jsonl(DATA_DIR / f"{args.split}.jsonl")
    print(f"Evaluating on {len(examples)} {args.split} examples ({tag})...")
    results = run_eval(extractor, examples, split_name=args.split)
    results["quantization"] = quant

    if args.split == "test":
        # Same metric on val, so the gap means something.
        val_examples = _load_jsonl(DATA_DIR / "val.jsonl")
        val = run_eval(extractor, val_examples, split_name="val", verbose=False)
        results["val_macro_f1"] = val["macro_f1"]
        results["generalization_gap"] = round(results["macro_f1"] - val["macro_f1"], 4)

    print("\nRunning adversarial eval...")
    adv = adversarial_eval(extractor)
    results["adversarial"] = adv

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    out_path = EXPERIMENTS_DIR / f"eval_{args.split}_{tag}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written -> {out_path}")

    if not args.no_mlflow:
        mlflow = setup_mlflow()
        with mlflow.start_run(run_name=f"eval_{args.split}_{tag}"):
            mlflow.log_metric(f"{args.split}_macro_f1", results["macro_f1"])
            mlflow.log_metric("p99_latency_ms", results["latency"]["p99_ms"])
            mlflow.log_metric("adversarial_pass_rate", adv["pass_rate"])
            if results["calibration"]["ece"] is not None:
                mlflow.log_metric("ece", results["calibration"]["ece"])
            for clause, m in results["per_clause"].items():
                if m["f1"] is not None:
                    mlflow.log_metric(f"{args.split}_f1_{clause.replace(' ', '_').replace('-', '_')}", m["f1"])
            if "generalization_gap" in results:
                mlflow.log_metric("generalization_gap", results["generalization_gap"])
            mlflow.log_artifact(str(out_path))

    print(f"\n{'=' * 72}\n  EVAL RESULTS ({args.split}, {tag})\n{'=' * 72}")
    print(f"  Macro F1 : {results['macro_f1']:.4f}   (micro {results['micro_f1']:.4f})")
    if "val_macro_f1" in results:
        print(f"  Val F1   : {results['val_macro_f1']:.4f}   (test - val: {results['generalization_gap']:+.4f})")
    print(f"\n  {'Clause':<28} {'F1':>6}  {'95% CI':<15} {'n+':>4} {'n-':>4} {'FPR-':>6}")
    for clause, m in results["per_clause"].items():
        if m["f1"] is None:
            print(f"  {clause:<28}   (no examples)")
            continue
        fpr = m["false_positive_rate_negative"]
        print(f"  {clause:<28} {m['f1']:.4f}  [{m['ci_95'][0]:.3f}, {m['ci_95'][1]:.3f}] "
              f"{m['n_positive']:>4} {m['n_negative']:>4} {'' if fpr is None else f'{fpr:.3f}':>6}")
    lat = results["latency"]
    print(f"\n  Latency: p50={lat['p50_ms']}ms  p95={lat['p95_ms']}ms  p99={lat['p99_ms']}ms "
          f"(target {P99_LATENCY_TARGET_MS}ms: {'MET' if lat['target_met'] else 'MISSED'})")
    print(f"  Calibration ECE: {results['calibration']['ece']}")
    print(f"  Adversarial pass rate: {adv['pass_rate']:.0%}\n{'=' * 72}")


if __name__ == "__main__":
    main()
