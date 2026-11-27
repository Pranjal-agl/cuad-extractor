"""
src/serving/serve.py

FastAPI serving endpoint for the CUAD clause extractor.

ENDPOINTS
---------
POST /extract
  Input:  {"text": str, "clause_types": list[str] | null, "min_score": float}
  Output: {"clauses":  {clause_type: [span_text, ...]},
           "spans":    {clause_type: [{"text", "start", "end", "score"}, ...]},
           "latency_ms", "n_windows", "p99_exceeded", "error"}
GET /health   model status + rolling p99 over the last 1000 requests
GET /metrics  latency percentiles and share of requests over target
GET /clauses  clause types the loaded model supports

CONFIGURATION (environment variables)
-------------------------------------
CUAD_MODEL_PATH       checkpoint dir (default: models/best_config.json)
CUAD_QUANTIZE         none | int8 (CPU) | nf4 (GPU)       default: none
CUAD_LOG_PREDICTIONS  1 to append predictions to data/production_log.jsonl
                      for src.eval.monitor                   default: 0
CUAD_MAX_CHARS        reject longer inputs with 413          default: 200000

DESIGN
------
- One model instance loaded at startup. Startup fails loudly if no model is
  found, rather than serving a broken API.
- All clause types come from a single forward pass per window; long texts are
  covered by overlapping sliding windows (see src.inference).
- /extract is a plain `def`, so FastAPI runs it in its threadpool and the
  event loop stays free for /health. A lock serialises model calls so
  concurrent requests don't oversubscribe CPU threads / GPU memory.
- Graceful degradation: if inference fails, the request returns empty spans
  with an error message instead of a 500.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.config import P99_LATENCY_TARGET_MS

LATENCY_WINDOW = 1000
MAX_CHARS = int(os.environ.get("CUAD_MAX_CHARS", "200000"))

log = logging.getLogger("cuad.serve")
_latency_window: deque = deque(maxlen=LATENCY_WINDOW)
_model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.inference import ClauseExtractor

    quantize = os.environ.get("CUAD_QUANTIZE", "none")
    app.state.extractor = ClauseExtractor.load(os.environ.get("CUAD_MODEL_PATH"), quantize=quantize)
    app.state.quantize = quantize
    app.state.log_predictions = os.environ.get("CUAD_LOG_PREDICTIONS", "0") == "1"
    app.state.extractor.extract("Warm-up request.")  # keep one-off init cost out of the first real request
    log.info("Loaded model (quantize=%s) on %s", quantize, app.state.extractor.device)
    yield


app = FastAPI(
    title="CUAD Clause Extractor",
    description="Extracts legal clause spans from contract text using fine-tuned DeBERTa-v3-base",
    lifespan=lifespan,
)


class ExtractRequest(BaseModel):
    text: str
    clause_types: list[str] | None = None  # None = every clause type the model knows
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="drop spans below this confidence")


class Span(BaseModel):
    text: str
    start: int
    end: int
    score: float | None = None


class ExtractResponse(BaseModel):
    clauses: dict[str, list[str]]
    spans: dict[str, list[Span]]
    latency_ms: float
    n_windows: int = 0
    p99_exceeded: bool
    error: str | None = None


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest, request: Request):
    extractor = request.app.state.extractor
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    if len(req.text) > MAX_CHARS:
        raise HTTPException(status_code=413, detail=f"text longer than {MAX_CHARS} characters")
    targets = req.clause_types or extractor.clause_types
    invalid = [c for c in targets if c not in extractor.clause_types]
    if invalid:
        raise HTTPException(status_code=400,
                            detail=f"Unknown clause types: {invalid}. Valid: {extractor.clause_types}")

    t0 = time.perf_counter()
    spans: dict[str, list[dict]] = {c: [] for c in targets}
    error_msg, n_windows = None, 0
    try:
        with _model_lock:
            out = extractor.extract(req.text, clause_types=targets)
        n_windows = out["n_windows"]
        for sp in out["spans"]:
            if sp.get("score") is None or sp["score"] >= req.min_score:
                spans[sp["clause"]].append(
                    {"text": sp["text"], "start": sp["start"], "end": sp["end"], "score": sp.get("score")})
    except Exception as e:  # graceful degradation
        log.exception("inference failed")
        error_msg = f"inference failed: {type(e).__name__}"
        spans = {c: [] for c in targets}

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    _latency_window.append(latency_ms)

    if request.app.state.log_predictions and error_msg is None:
        from src.eval.monitor import log_prediction

        for clause, items in spans.items():
            scores = [s["score"] for s in items if s["score"] is not None]
            log_prediction(req.text, clause, [s["text"] for s in items], latency_ms,
                           confidence=max(scores) if scores else None)

    return ExtractResponse(
        clauses={c: [s["text"] for s in items] for c, items in spans.items()},
        spans=spans,
        latency_ms=latency_ms,
        n_windows=n_windows,
        p99_exceeded=latency_ms > P99_LATENCY_TARGET_MS,
        error=error_msg,
    )


@app.get("/clauses")
def clauses(request: Request):
    return {"clause_types": request.app.state.extractor.clause_types}


@app.get("/health")
def health(request: Request):
    lats = list(_latency_window)
    p99 = float(np.percentile(lats, 99)) if lats else None
    extractor = getattr(request.app.state, "extractor", None)
    return {
        "status": "ok" if extractor is not None else "no_model",
        "model_loaded": extractor is not None,
        "quantize": getattr(request.app.state, "quantize", None),
        "device": str(extractor.device) if extractor is not None else None,
        "requests_tracked": len(lats),
        "p99_ms": round(p99, 2) if p99 is not None else None,
        "p99_target_ms": P99_LATENCY_TARGET_MS,
        "p99_target_met": (p99 <= P99_LATENCY_TARGET_MS) if p99 is not None else None,
    }


@app.get("/metrics")
def metrics():
    lats = list(_latency_window)
    if not lats:
        return {"message": "No requests yet"}
    p99 = float(np.percentile(lats, 99))
    return {
        "n_requests": len(lats),
        "p50_ms": round(float(np.percentile(lats, 50)), 2),
        "p95_ms": round(float(np.percentile(lats, 95)), 2),
        "p99_ms": round(p99, 2),
        "mean_ms": round(float(np.mean(lats)), 2),
        "p99_target_ms": P99_LATENCY_TARGET_MS,
        "p99_target_met": p99 <= P99_LATENCY_TARGET_MS,
        "pct_exceeding_target": round(100 * sum(1 for x in lats if x > P99_LATENCY_TARGET_MS) / len(lats), 1),
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("src.serving.serve:app", host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", "8000")), reload=False)
