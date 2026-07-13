"""
src/training/quantize.py

Quantizes the best fine-tuned checkpoint to 4-bit NF4 via bitsandbytes,
measures accuracy drop, latency gain, and memory saving vs FP32.
"""

import json
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
EXPERIMENTS_DIR = ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

P99_LATENCY_TARGET_MS = 150
N_WARMUP = 10
N_TIMING = 100


def load_best_model_path() -> Path:
    config_path = MODELS_DIR / "best_config.json"
    if not config_path.exists():
        raise FileNotFoundError("Run train.py first to produce best_config.json")
    with open(config_path) as f:
        return Path(json.load(f)["model_path"])


def load_fp32_model(model_path: Path):
    from transformers import AutoModelForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    return model


def load_4bit_model(model_path: Path):
    from transformers import AutoModelForTokenClassification, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto",
    )
    model.eval()
    return model


def measure_latency(model, tokenizer, text, n_warmup=N_WARMUP, n_timing=N_TIMING, device="cpu"):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=False)
    if not next(model.parameters()).is_cuda:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        for _ in range(n_warmup):
            model(**inputs)
    times = []
    with torch.no_grad():
        for _ in range(n_timing):
            t0 = time.perf_counter()
            model(**inputs)
            times.append((time.perf_counter() - t0) * 1000)
    return {
        "p50_ms": round(float(np.percentile(times, 50)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
        "p99_ms": round(float(np.percentile(times, 99)), 2),
        "mean_ms": round(float(np.mean(times)), 2),
    }


def measure_f1_on_val(model, tokenizer, n_samples=200):
    import json
    from src.training.train import compute_f1, tokenize_and_label

    val_path = DATA_DIR / "val.jsonl"
    with open(val_path) as f:
        examples = [json.loads(l) for l in f if l.strip()][:n_samples]

    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for ex in examples:
            enc = tokenize_and_label(ex, tokenizer)
            input_ids = torch.tensor([enc["input_ids"]]).to(device)
            attention_mask = torch.tensor([enc["attention_mask"]]).to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = out.logits.argmax(dim=-1).cpu().numpy().flatten().tolist()
            all_preds.extend(preds)
            all_labels.extend(enc["labels"])
    return compute_f1(all_preds, all_labels)["macro"]


def model_size_mb(model_path: Path) -> float:
    total = sum(p.stat().st_size for p in model_path.rglob("*") if p.is_file())
    return round(total / 1024 / 1024, 1)


def main():
    model_path = load_best_model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_text = (
        "This Agreement shall be governed by and construed in accordance with "
        "the laws of the State of Delaware, without regard to its conflict of "
        "law provisions. Either party may terminate this Agreement for convenience "
        "upon thirty (30) days written notice to the other party."
    )

    print("Loading FP32 model...")
    fp32_model = load_fp32_model(model_path).to(device)
    fp32_latency = measure_latency(fp32_model, tokenizer, sample_text, device=device)
    fp32_f1 = measure_f1_on_val(fp32_model, tokenizer)
    fp32_size = model_size_mb(model_path)
    print(f"FP32: F1={fp32_f1:.4f}  p99={fp32_latency['p99_ms']}ms  size={fp32_size}MB")
    del fp32_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    quant_f1 = quant_latency = quant_size = None

    if torch.cuda.is_available():
        print("Loading 4-bit quantized model...")
        quant_model = load_4bit_model(model_path)
        quant_latency = measure_latency(quant_model, tokenizer, sample_text)
        quant_f1 = measure_f1_on_val(quant_model, tokenizer)
        quant_path = MODELS_DIR / "quantized_4bit"
        quant_path.mkdir(exist_ok=True)
        quant_model.save_pretrained(quant_path)
        tokenizer.save_pretrained(quant_path)
        quant_size = model_size_mb(quant_path)
        print(f"4-bit: F1={quant_f1:.4f}  p99={quant_latency['p99_ms']}ms  size={quant_size}MB")
        del quant_model
        torch.cuda.empty_cache()
    else:
        print("No CUDA available -- skipping 4-bit quantization (requires GPU).")

    report = {
        "fp32": {"val_macro_f1": fp32_f1, "latency": fp32_latency, "model_size_mb": fp32_size},
        "quantized_4bit": {
            "val_macro_f1": quant_f1, "latency": quant_latency, "model_size_mb": quant_size,
            "f1_drop": round(fp32_f1 - quant_f1, 4) if quant_f1 else None,
            "latency_speedup_p99": round(fp32_latency["p99_ms"] / quant_latency["p99_ms"], 2) if quant_latency else None,
        },
        "p99_target_ms": P99_LATENCY_TARGET_MS,
        "p99_target_met_fp32": fp32_latency["p99_ms"] <= P99_LATENCY_TARGET_MS,
        "p99_target_met_4bit": (quant_latency["p99_ms"] <= P99_LATENCY_TARGET_MS) if quant_latency else None,
    }

    report_path = EXPERIMENTS_DIR / "quantization_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written -> {report_path}")

    from src.training.train import EXPERIMENT_NAME
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="quantization"):
        mlflow.log_params({"p99_target_ms": P99_LATENCY_TARGET_MS})
        mlflow.log_metric("fp32_val_f1", fp32_f1)
        mlflow.log_metric("fp32_p99_ms", fp32_latency["p99_ms"])
        if quant_f1:
            mlflow.log_metric("quant4bit_val_f1", quant_f1)
            mlflow.log_metric("quant4bit_p99_ms", quant_latency["p99_ms"])
            mlflow.log_metric("f1_drop", fp32_f1 - quant_f1)
            mlflow.log_metric("latency_speedup_p99", report["quantized_4bit"]["latency_speedup_p99"])
        mlflow.log_artifact(str(report_path))

    print("\n" + "="*60)
    print("  QUANTIZATION SUMMARY")
    print("="*60)
    print(f"  FP32   F1={fp32_f1:.4f}  p99={fp32_latency['p99_ms']}ms  {fp32_size}MB")
    if quant_f1:
        drop = fp32_f1 - quant_f1
        speedup = report["quantized_4bit"]["latency_speedup_p99"]
        print(f"  4-bit  F1={quant_f1:.4f}  p99={quant_latency['p99_ms']}ms  {quant_size}MB")
        print(f"  Drop: {drop:.4f} F1  |  Speedup: {speedup}x p99")
    target_met = "YES" if report["p99_target_met_fp32"] else "NO"
    print(f"  p99 target ({P99_LATENCY_TARGET_MS}ms) met by FP32: {target_met}")
    print("="*60)


if __name__ == "__main__":
    main()
