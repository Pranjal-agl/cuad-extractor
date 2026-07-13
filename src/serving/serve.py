"""
src/serving/serve.py

FastAPI serving endpoint for the CUAD clause extractor.

ENDPOINTS
---------
POST /extract
  Input:  {"text": str, "clause_types": list[str] | null}
  Output: {"clauses": {clause_type: [span, ...]}, "latency_ms": float}

GET /health
  Returns model status and p99 latency from the last 1000 requests.

GET /metrics
  Returns full latency histogram and per-clause request counts.

DESIGN
------
- Single model instance, loaded once at startup
- Per-request latency tracked in a rolling window (last 1000)
- Requests exceeding the p99 target are flagged in the response
- Graceful degradation: if model fails, returns empty spans with error flag
  rather than 500

LOAD TEST
---------
Run load_test.py to simulate concurrent requests and get real p99 numbers.
"""

import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent.parent

CLAUSE_TYPES = [
    "Governing Law", "Termination For Convenience", "Renewal Term",
    "Non-Compete", "Confidentiality Of Agreements", "Indemnification",
    "Cap On Liability", "Assignment",
]
P99_TARGET_MS = 150
LATENCY_WINDOW = 1000

# Global model state
_model = None
_tokenizer = None
_latency_window: deque = deque(maxlen=LATENCY_WINDOW)


def _load_model():
    global _model, _tokenizer
    import json
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from pathlib import Path

    models_dir = ROOT / "models"
    quant_path = models_dir / "quantized_4bit"
    best_path_file = models_dir / "best_config.json"

    if quant_path.exists() and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        _model = AutoModelForTokenClassification.from_pretrained(
            quant_path, quantization_config=bnb, device_map="auto"
        )
        _tokenizer = AutoTokenizer.from_pretrained(quant_path)
        print("Loaded 4-bit quantized model")
    elif best_path_file.exists():
        with open(best_path_file) as f:
            model_path = Path(json.load(f)["model_path"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Loaded FP32 model on {device}")
    else:
        raise RuntimeError("No trained model found. Run train.py first.")

    _model.eval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="CUAD Clause Extractor",
    description="Extracts legal clause spans from contract text using fine-tuned DeBERTa-v3-base",
    lifespan=lifespan,
)


class ExtractRequest(BaseModel):
    text: str
    clause_types: Optional[list[str]] = None  # None = extract all 8 types


class ExtractResponse(BaseModel):
    clauses: dict[str, list[str]]
    latency_ms: float
    p99_exceeded: bool
    error: Optional[str] = None


def _extract_spans(text: str, clause: str) -> list[str]:
    from src.eval.evaluate import predict
    spans, _ = predict(_model, _tokenizer, text, clause)
    return spans


@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    target_clauses = req.clause_types or CLAUSE_TYPES
    invalid = [c for c in target_clauses if c not in CLAUSE_TYPES]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown clause types: {invalid}. Valid: {CLAUSE_TYPES}"
        )

    t0 = time.perf_counter()
    result_clauses: dict[str, list[str]] = {}
    error_msg = None

    try:
        for clause in target_clauses:
            result_clauses[clause] = _extract_spans(req.text, clause)
    except Exception as e:
        error_msg = str(e)
        result_clauses = {c: [] for c in target_clauses}

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    _latency_window.append(latency_ms)

    return ExtractResponse(
        clauses=result_clauses,
        latency_ms=latency_ms,
        p99_exceeded=latency_ms > P99_TARGET_MS,
        error=error_msg,
    )


@app.get("/health")
async def health():
    lats = list(_latency_window)
    return {
        "status": "ok" if _model is not None else "no_model",
        "model_loaded": _model is not None,
        "requests_tracked": len(lats),
        "p99_ms": round(float(np.percentile(lats, 99)), 2) if lats else None,
        "p99_target_ms": P99_TARGET_MS,
        "p99_target_met": (
            float(np.percentile(lats, 99)) <= P99_TARGET_MS
        ) if lats else None,
    }


@app.get("/metrics")
async def metrics():
    lats = list(_latency_window)
    if not lats:
        return {"message": "No requests yet"}
    return {
        "n_requests": len(lats),
        "p50_ms": round(float(np.percentile(lats, 50)), 2),
        "p95_ms": round(float(np.percentile(lats, 95)), 2),
        "p99_ms": round(float(np.percentile(lats, 99)), 2),
        "mean_ms": round(float(np.mean(lats)), 2),
        "p99_target_ms": P99_TARGET_MS,
        "p99_target_met": float(np.percentile(lats, 99)) <= P99_TARGET_MS,
        "pct_exceeding_target": round(
            100 * sum(1 for l in lats if l > P99_TARGET_MS) / len(lats), 1
        ),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.serve:app", host="0.0.0.0", port=8000, reload=False)
