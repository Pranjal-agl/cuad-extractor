"""
src/training/quantize.py

Measures the accuracy / latency / size trade-off of two quantization schemes
against the FP32 checkpoint, each on the hardware it is meant for:

  int8  PyTorch dynamic INT8 quantization of all nn.Linear layers.
        CPU-only, no extra dependencies, applied at load time.
        Compared against FP32 *on CPU*.
  nf4   bitsandbytes 4-bit NF4. GPU-only. Compared against FP32 *on GPU*.
        Previously crashed on load with the pinned transformers 4.44.2 +
        recent accelerate; see src.inference.allow_noop_to_for_bnb for the
        root cause and the workaround.

F1 is the same span-level macro F1 used by train.py (selection) and
evaluate.py, measured on (a prefix of) the validation split.

    python -m src.training.quantize                # all schemes available here
    python -m src.training.quantize --n-samples 0  # latency/size only
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import traceback

import torch

from src.config import DATA_DIR, EXPERIMENTS_DIR, MODELS_DIR, P99_LATENCY_TARGET_MS, setup_mlflow
from src.inference import ClauseExtractor, load_model, resolve_model_path
from src.metrics import latency_summary

N_WARMUP = 10
N_TIMING = 100
SAMPLE_TEXT = (
    "This Agreement shall be governed by and construed in accordance with "
    "the laws of the State of Delaware, without regard to its conflict of "
    "law provisions. Either party may terminate this Agreement for convenience "
    "upon thirty (30) days written notice to the other party."
)


def measure_latency(extractor: ClauseExtractor, text: str = SAMPLE_TEXT,
                    n_warmup: int = N_WARMUP, n_timing: int = N_TIMING) -> dict:
    for _ in range(n_warmup):
        extractor.extract(text)
    return latency_summary([extractor.extract(text)["latency_ms"] for _ in range(n_timing)],
                           P99_LATENCY_TARGET_MS)


def val_f1(extractor: ClauseExtractor, n_samples: int) -> float | None:
    if n_samples == 0:
        return None
    from src.eval.evaluate import run_eval

    with open(DATA_DIR / "val.jsonl") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    if n_samples > 0:
        examples = examples[:n_samples]
    return run_eval(extractor, examples, split_name="val", verbose=False)["macro_f1"]


def dir_size_mb(path) -> float:
    weights = [p for p in path.rglob("*") if p.is_file() and p.suffix in {".safetensors", ".bin"}]
    files = weights or [p for p in path.rglob("*") if p.is_file()]
    return round(sum(p.stat().st_size for p in files) / 2**20, 1)


def state_dict_size_mb(model) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return round(buf.tell() / 2**20, 1)


def _drop(base, new):
    return round(base - new, 4) if base is not None and new is not None else None


def _ratio(a, b):
    return round(a / b, 2) if a and b else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--n-samples", type=int, default=200, help="val examples for F1 (-1 = all, 0 = skip)")
    parser.add_argument("--n-timing", type=int, default=N_TIMING)
    parser.add_argument("--schemes", default="int8,nf4")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    schemes = set(args.schemes.split(","))
    model_path = resolve_model_path(args.model_path)
    has_gpu = torch.cuda.is_available()
    report: dict = {"model_path": str(model_path), "p99_target_ms": P99_LATENCY_TARGET_MS}

    # FP32 baseline: F1 once (device does not change the math), latency per device.
    print("FP32 baseline...")
    fp32_cpu = ClauseExtractor(*load_model(model_path, device="cpu"))
    fp32 = {"val_macro_f1": None, "model_size_mb": dir_size_mb(model_path),
            "latency_cpu": measure_latency(fp32_cpu, n_timing=args.n_timing)}
    if has_gpu:
        fp32_gpu = ClauseExtractor(*load_model(model_path, device="cuda"))
        fp32["val_macro_f1"] = val_f1(fp32_gpu, args.n_samples)
        fp32["latency_gpu"] = measure_latency(fp32_gpu, n_timing=args.n_timing)
        del fp32_gpu
        torch.cuda.empty_cache()
    else:
        fp32["val_macro_f1"] = val_f1(fp32_cpu, args.n_samples)
    del fp32_cpu
    report["fp32"] = fp32
    print(f"  F1={fp32['val_macro_f1']}  CPU p99={fp32['latency_cpu']['p99_ms']}ms"
          + (f"  GPU p99={fp32['latency_gpu']['p99_ms']}ms" if has_gpu else "")
          + f"  size={fp32['model_size_mb']}MB")

    if "int8" in schemes:
        print("INT8 dynamic (CPU)...")
        q = ClauseExtractor(*load_model(model_path, quantize="int8"))
        f1 = val_f1(q, args.n_samples)
        lat = measure_latency(q, n_timing=args.n_timing)
        report["int8_dynamic_cpu"] = {
            "val_macro_f1": f1, "latency_cpu": lat, "model_size_mb": state_dict_size_mb(q.model),
            "f1_drop": _drop(fp32["val_macro_f1"], f1),
            "cpu_speedup_p99": _ratio(fp32["latency_cpu"]["p99_ms"], lat["p99_ms"]),
        }
        print(f"  F1={f1}  CPU p99={lat['p99_ms']}ms  size={report['int8_dynamic_cpu']['model_size_mb']}MB")
        del q

    if "nf4" in schemes:
        if not has_gpu:
            report["nf4_gpu"] = {"skipped": "no CUDA device"}
            print("NF4: skipped (no GPU)")
        else:
            print("NF4 4-bit (GPU)...")
            try:
                q = ClauseExtractor(*load_model(model_path, quantize="nf4"))
                f1 = val_f1(q, args.n_samples)
                lat = measure_latency(q, n_timing=args.n_timing)
                out_dir = MODELS_DIR / "quantized_4bit"
                shutil.rmtree(out_dir, ignore_errors=True)
                q.model.save_pretrained(out_dir)
                q.tokenizer.save_pretrained(out_dir)
                report["nf4_gpu"] = {
                    "val_macro_f1": f1, "latency_gpu": lat, "model_size_mb": dir_size_mb(out_dir),
                    "f1_drop": _drop(fp32["val_macro_f1"], f1),
                    "gpu_speedup_p99": _ratio(fp32["latency_gpu"]["p99_ms"], lat["p99_ms"]),
                }
                print(f"  F1={f1}  GPU p99={lat['p99_ms']}ms  size={report['nf4_gpu']['model_size_mb']}MB")
                del q
                torch.cuda.empty_cache()
            except Exception as e:  # keep the rest of the report
                traceback.print_exc()
                report["nf4_gpu"] = {"error": f"{type(e).__name__}: {e}"}

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    report_path = EXPERIMENTS_DIR / "quantization_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written -> {report_path}")

    if not args.no_mlflow:
        mlflow = setup_mlflow()
        with mlflow.start_run(run_name="quantization"):
            for scheme, r in report.items():
                if isinstance(r, dict):
                    for k in ("val_macro_f1", "f1_drop", "cpu_speedup_p99", "gpu_speedup_p99", "model_size_mb"):
                        if isinstance(r.get(k), (int, float)):
                            mlflow.log_metric(f"{scheme}_{k}", r[k])
            mlflow.log_artifact(str(report_path))


if __name__ == "__main__":
    main()
