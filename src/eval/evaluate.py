"""
src/eval/evaluate.py

Rigorous eval harness for the CUAD clause extractor: per-clause F1 with
95% Wilson CI, precision/recall, adversarial cases, and overfit check
(val F1 vs test F1).
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
EXPERIMENTS_DIR = ROOT / "experiments"

CLAUSE_TYPES = [
    "Governing Law",
    "Termination For Convenience",
    "Renewal Term",
    "Non-Compete",
    "Cap On Liability",
    "Anti-Assignment",
    "Audit Rights",
    "Exclusivity",
]

P99_LATENCY_TARGET_MS = 150


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    p = correct / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    spread = (z * (p * (1-p)/total + z**2/(4*total**2))**0.5) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def load_model(use_quantized: bool = False):
    quant_path = MODELS_DIR / "quantized_4bit"
    best_path_file = MODELS_DIR / "best_config.json"

    if use_quantized and quant_path.exists():
        print("Loading 4-bit quantized model...")
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            quant_path, quantization_config=bnb,
        )
        tokenizer = AutoTokenizer.from_pretrained(quant_path)
    else:
        with open(best_path_file) as f:
            model_path = Path(json.load(f)["model_path"])
        print(f"Loading FP32 model from {model_path}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()
    return model, tokenizer


def predict(model, tokenizer, text: str, clause: str) -> tuple[list[str], float]:
    from src.training.train import ID2LABEL

    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs)
    latency_ms = (time.perf_counter() - t0) * 1000

    pred_ids = out.logits.argmax(dim=-1)[0].tolist()

    spans = []
    current_span_start = None
    current_span_chars = []

    for idx, (pred_id, (tok_start, tok_end)) in enumerate(zip(pred_ids, offsets)):
        label = ID2LABEL.get(pred_id, "O")
        label_clause = label.split("-", 1)[-1] if label != "O" else "O"

        if label.startswith("B-") and label_clause == clause:
            if current_span_start is not None:
                spans.append(text[current_span_start:current_span_chars[-1]])
            current_span_start = tok_start
            current_span_chars = [tok_end]
        elif label.startswith("I-") and label_clause == clause and current_span_start is not None:
            current_span_chars.append(tok_end)
        else:
            if current_span_start is not None:
                spans.append(text[current_span_start:current_span_chars[-1]])
                current_span_start = None
                current_span_chars = []

    if current_span_start is not None:
        spans.append(text[current_span_start:current_span_chars[-1]])

    return spans, latency_ms


def token_f1(pred_spans: list[str], gold_spans: list[str], context: str) -> dict[str, float]:
    def to_token_set(spans: list[str]) -> set[int]:
        chars = set()
        for span in spans:
            start = context.find(span)
            if start >= 0:
                chars.update(range(start, start + len(span)))
        return chars

    pred_chars = to_token_set(pred_spans)
    gold_chars = to_token_set(gold_spans)

    if not pred_chars and not gold_chars:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_chars or not gold_chars:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_chars & gold_chars)
    precision = tp / len(pred_chars)
    recall = tp / len(gold_chars)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def run_eval(model, tokenizer, examples: list[dict], split_name: str = "test") -> dict:
    per_clause_scores: dict[str, list[dict]] = defaultdict(list)
    all_latencies: list[float] = []
    errors: list[dict] = []

    for i, ex in enumerate(examples):
        pred_spans, latency_ms = predict(model, tokenizer, ex["context"], ex["clause"])
        gold_spans = ex["answers"]
        all_latencies.append(latency_ms)

        scores = token_f1(pred_spans, gold_spans, ex["context"])
        per_clause_scores[ex["clause"]].append(scores)

        if scores["f1"] < 0.5 and gold_spans:
            errors.append({
                "clause": ex["clause"], "context_snippet": ex["context"][:200],
                "gold": gold_spans[:1], "pred": pred_spans[:1], "f1": round(scores["f1"], 3),
            })

        if (i + 1) % 50 == 0:
            print(f"  [{split_name}] {i+1}/{len(examples)}")

    results: dict[str, dict] = {}
    all_f1s = []
    for clause in CLAUSE_TYPES:
        scores_list = per_clause_scores.get(clause, [])
        if not scores_list:
            results[clause] = {"f1": None, "precision": None, "recall": None, "n": 0}
            continue

        f1s = [s["f1"] for s in scores_list]
        precs = [s["precision"] for s in scores_list]
        recs = [s["recall"] for s in scores_list]
        mean_f1 = float(np.mean(f1s))
        all_f1s.append(mean_f1)

        correct = sum(1 for f in f1s if f > 0.5)
        ci_lo, ci_hi = wilson_ci(correct, len(f1s))

        results[clause] = {
            "f1": round(mean_f1, 4), "precision": round(float(np.mean(precs)), 4),
            "recall": round(float(np.mean(recs)), 4), "n": len(f1s),
            "n_positive": sum(1 for ex in examples if ex["clause"] == clause and ex["has_answer"]),
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        }

    macro_f1 = round(float(np.mean(all_f1s)), 4) if all_f1s else 0.0

    latency_summary = {
        "p50_ms": round(float(np.percentile(all_latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(all_latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(all_latencies, 99)), 2),
        "mean_ms": round(float(np.mean(all_latencies)), 2),
        "target_ms": P99_LATENCY_TARGET_MS,
        "target_met": float(np.percentile(all_latencies, 99)) <= P99_LATENCY_TARGET_MS,
    }

    return {
        "split": split_name, "macro_f1": macro_f1, "per_clause": results,
        "latency": latency_summary, "n_examples": len(examples),
        "n_errors_collected": len(errors), "errors": errors[:20],
    }


def adversarial_eval(model, tokenizer) -> dict:
    cases = [
        {"name": "truncated_context",
         "context": "This Agreement shall be governed by the laws of California.",
         "clause": "Governing Law", "answers": ["laws of California"], "has_answer": True},
        {"name": "no_clause_present",
         "context": "The parties agree to the terms and conditions set forth herein. "
                     "Payment shall be made within 30 days of invoice receipt.",
         "clause": "Governing Law", "answers": [], "has_answer": False},
        {"name": "multiple_spans",
         "context": "This Agreement is governed by Delaware law. For disputes arising "
                     "outside the US, New York law shall apply.",
         "clause": "Governing Law", "answers": ["Delaware law", "New York law"], "has_answer": True},
        {"name": "boilerplate_noise",
         "context": "IN WITNESS WHEREOF, the parties hereto have executed this Agreement "
                     "as of the date first written above. COMPANY A By: Name: Title: Date: "
                     "COMPANY B By: Name: Title: Date:",
         "clause": "Cap On Liability", "answers": [], "has_answer": False},
    ]

    results = []
    for case in cases:
        pred_spans, latency_ms = predict(model, tokenizer, case["context"], case["clause"])
        scores = token_f1(pred_spans, case["answers"], case["context"])
        results.append({
            "case": case["name"], "clause": case["clause"], "expected": case["answers"],
            "predicted": pred_spans, "f1": round(scores["f1"], 3), "latency_ms": round(latency_ms, 2),
            "pass": scores["f1"] >= 0.5 if case["has_answer"] else len(pred_spans) == 0,
        })
        status = "PASS" if results[-1]["pass"] else "FAIL"
        print(f"  [{status}] {case['name']}: f1={scores['f1']:.3f}")

    return {"adversarial_cases": results, "pass_rate": sum(r["pass"] for r in results) / len(results)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    from src.training.prepare_data import verify_lock
    verify_lock()

    model, tokenizer = load_model(use_quantized=args.quantized)

    with open(DATA_DIR / f"{args.split}.jsonl") as f:
        examples = [json.loads(l) for l in f if l.strip()]
    print(f"Evaluating on {len(examples)} {args.split} examples...")

    results = run_eval(model, tokenizer, examples, split_name=args.split)

    print("\nRunning adversarial eval...")
    adv_results = adversarial_eval(model, tokenizer)
    results["adversarial"] = adv_results

    if args.split == "test":
        best_config_path = MODELS_DIR / "best_config.json"
        if best_config_path.exists():
            with open(best_config_path) as f:
                best = json.load(f)
            results["val_macro_f1"] = best["val_f1"]
            results["overfit_gap"] = round(best["val_f1"] - results["macro_f1"], 4)

    out_path = EXPERIMENTS_DIR / f"eval_{args.split}_{'quant' if args.quantized else 'fp32'}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written -> {out_path}")

    from src.training.train import EXPERIMENT_NAME
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = f"eval_{args.split}_{'quant' if args.quantized else 'fp32'}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metric(f"{args.split}_macro_f1", results["macro_f1"])
        mlflow.log_metric("p99_latency_ms", results["latency"]["p99_ms"])
        mlflow.log_metric("adversarial_pass_rate", adv_results["pass_rate"])
        for clause, m in results["per_clause"].items():
            if m["f1"] is not None:
                safe = clause.replace(" ", "_")
                mlflow.log_metric(f"{args.split}_f1_{safe}", m["f1"])
        if "overfit_gap" in results:
            mlflow.log_metric("overfit_gap", results["overfit_gap"])
        mlflow.log_artifact(str(out_path))

    print(f"\n{'='*60}")
    print(f"  EVAL RESULTS ({args.split}, {'4-bit' if args.quantized else 'FP32'})")
    print(f"{'='*60}")
    print(f"  Macro F1 : {results['macro_f1']:.4f}")
    if "val_macro_f1" in results:
        print(f"  Val F1   : {results['val_macro_f1']:.4f}  (overfit gap: {results['overfit_gap']:+.4f})")
    print(f"\n  Per-clause F1:")
    for clause, m in results["per_clause"].items():
        if m["f1"] is not None:
            ci = m["ci_95"]
            print(f"    {clause:<35} {m['f1']:.4f}  CI=[{ci[0]:.3f}, {ci[1]:.3f}]  n={m['n']}")
    print(f"\n  Latency: p50={results['latency']['p50_ms']}ms  p95={results['latency']['p95_ms']}ms  p99={results['latency']['p99_ms']}ms")
    target_str = "MET" if results["latency"]["target_met"] else "MISSED"
    print(f"  p99 target ({P99_LATENCY_TARGET_MS}ms): {target_str}")
    print(f"\n  Adversarial pass rate: {adv_results['pass_rate']:.0%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
