# CUAD Clause Extractor

[![CI](https://github.com/Pranjal-agl/cuad-extractor/actions/workflows/ci.yml/badge.svg)](https://github.com/Pranjal-agl/cuad-extractor/actions/workflows/ci.yml)

Fine-tuned DeBERTa-v3-base for extractive clause identification in legal contracts. Given contract text of any length, the model tags character spans for 8 CUAD clause types — **Governing Law, Termination For Convenience, Renewal Term, Non-Compete, Cap On Liability, Anti-Assignment, Audit Rights, Exclusivity** — in a single pass, each with a confidence score.

Built to cover what a pure agentic pipeline cannot: model ownership, quantization with a measured accuracy/latency trade-off, validation-driven hyperparameter search tracked in MLflow, and a p99 latency target verified under load.

---

## Results

> **Status:** the numbers below come from the 27 Jul 2026 Kaggle run of the *first* version of this pipeline (`notebooks/kaggle_train.ipynb`). Review of that version found bugs that affect how they should be read (next section). The fixes are in the code; the notebook needs a re-run to produce numbers for the current pipeline.

| Metric (27 Jul run) | FP32 |
|---|---|
| Test macro F1 (span-level, 5 clause types) | 0.7747 |
| Val macro F1 reported at selection (token-level, ÷8) | 0.4318 |
| p99 latency (single window, T4) | 29.1 ms |
| Model size | 711.9 MB |
| Adversarial pass rate | 50% (2/4) |

| Clause (test) | F1 | n |
|---|---|---|
| Governing Law | 0.973 | 67 |
| Renewal Term | 0.764 | 30 |
| Non-Compete | 0.754 | 27 |
| Cap On Liability | 0.729 | 109 |
| Termination For Convenience | 0.655 | 36 |

The original table also had "95% CIs". Those were Wilson intervals for *the fraction of examples with F1 > 0.5*, not for F1, so they're omitted here. The current code reports bootstrap CIs of mean F1.

### What the review found in that run

1. **Only 5 of the 8 clause types had any data.** `prepare_data.py` asked for "Confidentiality Of Agreements", "Indemnification" and "Assignment", which are not CUAD category names, so they matched nothing. Meanwhile `train.py`, `evaluate.py`, `serve.py` and `monitor.py` each kept *different* clause lists. Now there is one list in `src/config.py`, using real CUAD names, and `prepare_data` aborts if any configured type yields zero examples.
2. **The "test F1 is higher than val F1" gap was a metric artifact, not sampling noise.** Val F1 (used for model selection) was token-level and averaged over all 8 configured clauses, including the 3 with no data (F1 = 0). Test F1 was span-level and averaged over the 5 present clauses. 0.4318 × 8/5 = 0.691. Now the same span-level metric is used for selection, for val and for test, and `evaluate.py --split test` reports `test − val` on identical footing.
3. **Every example was positive.** The model never saw a window *without* the clause it was asked about, so "no clause here" was barely tested. Now negative windows are sampled from contracts that lack the clause (CUAD reviewers labelled every contract for every category).
4. **Other clauses inside a window were labelled `O`.** Each window only labelled its anchor annotation. A second Governing Law sentence, or a Cap On Liability clause in the same window, was taught as "not a clause". Now every annotation overlapping a window is labelled.
5. **Truncation.** Text past 384/512 tokens was dropped at train *and* inference time. Labels for spans crossing the cut were lost entirely, and the API couldn't see clauses beyond the first ~2 pages. Now training and inference both use overlapping sliding windows (384 tokens, 128 stride), and predictions are merged across windows.
6. **Train/inference mismatch.** Training used 384-token windows, while evaluation and serving used 512.
7. **Offsets were off by a few characters.** `context.strip()` and `answer.strip()` were applied without adjusting `start_in_context`. Offsets are now trimmed instead of the text.
8. **Over-correction for class imbalance.** Inverse-frequency weights fixed the all-`O` collapse, but the model then predicted 2.3× more clause tokens than exist (34,335 vs 14,946 on val), and precision trails recall on every clause. The grid now includes a `sqrt_inverse` weighting run as an ablation.
9. **Two adversarial "failures" were wrong gold labels.** CUAD annotators mark whole clause sentences, but the cases expected short phrases ("laws of California"), so a correct full-sentence prediction scored F1 0.47. The cases now follow the CUAD convention. Two new cases were added: a wrong-clause query, and a clause that only appears after the first window.

---

## Architecture

```
Contract text (any length)
        |
        v
DeBERTa-v3 tokenizer -> overlapping windows (384 tokens, stride 128)
        |
        v
DeBERTa-v3-base encoder + token classification head
(17 labels: O + B/I for 8 clause types; batched over windows)
        |
        v
BIO decoding per window (lenient: orphan I- starts a span)
        |
        v
Merge overlapping spans across windows -> {clause, start, end, text, score}
```

One forward pass per window produces every clause type at once. The original server ran the model once per requested clause type, up to 8× the work. The same `ClauseExtractor` (`src/inference.py`) is used by validation during training, evaluation, quantization and serving, so every number is produced the same way.

---

## What the doc required and where it is

| Requirement | Implementation |
|---|---|
| Held-out eval, no leakage | Contract-level split; `splits.lock` fingerprints test contracts **and** test examples and clause list |
| Reproducible runs | seed=42 everywhere (incl. data loader), MLflow logs params + git commit |
| Ablations isolating component contribution | 7-config grid incl. class-weighting ablation, all logged to MLflow |
| Validation-driven hyperparameter search | Selection on span-level val macro F1, test touched once |
| Honest overfit check | `evaluate.py --split test` scores val with the same metric and reports `test − val` |
| Quantization with trade-off measured | INT8 dynamic (CPU) vs FP32-on-CPU; NF4 4-bit (GPU) vs FP32-on-GPU; F1 drop, p99 speedup, size |
| p99 latency target | 150 ms; measured after warm-up in eval, quantization and the load test |
| Slice and adversarial evals | Per-clause F1 on positives, false-positive rate on negatives, 6 adversarial cases |
| Confidence intervals | 95% bootstrap CI of mean F1 per clause; Wilson CI for hit rate |
| Calibration | Reliability table + ECE of span confidence scores (`src/eval/calibration.py`) |
| Drift detection | `monitor.py`: prediction rate, text length, latency drift; server logs when `CUAD_LOG_PREDICTIONS=1` |
| Online/offline metric gap | `monitor.py --label` (shows predicted spans) + `--report` |
| Graceful degradation | Inference errors return empty spans + error flag, never a 500 |
| Tests / CI | `pytest` (35 tests, tiny local model, no downloads) + `ruff` on GitHub Actions |
| Model card | `MODEL_CARD.md` |

---

## Setup

```bash
git clone https://github.com/Pranjal-agl/cuad-extractor
cd cuad-extractor
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # add -dev for pytest/ruff
```

A GPU is recommended for training (Kaggle T4 / Colab). Serving and evaluation run fine on CPU, and `CUAD_QUANTIZE=int8` makes CPU serving faster. `notebooks/kaggle_train.ipynb` runs the whole pipeline on Kaggle.

## Running

```bash
# 1. Data: CUAD from HuggingFace -> data/{train,val,test}.jsonl + splits.lock
python -m src.training.prepare_data                 # --neg-ratio 0 to disable negatives

# 2. Train: grid search, everything logged to MLflow
python -m src.training.train                        # --configs 3,7  --prune
mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db

# 3. Quantization trade-off -> experiments/quantization_report.json
python -m src.training.quantize                     # --schemes int8,nf4

# 4. Evaluate once, after selection is final -> experiments/eval_test_<tag>.json
python -m src.eval.evaluate --split test            # --quantize int8|nf4
python -m src.eval.calibration --eval-file experiments/eval_test_fp32.json

# 5. Serve
python -m src.serving.serve                         # env: CUAD_QUANTIZE, CUAD_MODEL_PATH, ...

# 6. Load test (server must be running)
python -m src.serving.load_test --concurrency 4 --n-requests 200

# 7. Monitoring (serve with CUAD_LOG_PREDICTIONS=1)
python -m src.eval.monitor --label
python -m src.eval.monitor --report
```

### API

```bash
curl -X POST http://localhost:8000/extract -H "Content-Type: application/json" \
  -d '{"text": "This Agreement is governed by the laws of Delaware.", "clause_types": ["Governing Law"], "min_score": 0.5}'
```

```json
{
  "clauses": {"Governing Law": ["This Agreement is governed by the laws of Delaware."]},
  "spans":   {"Governing Law": [{"text": "This Agreement is governed by the laws of Delaware.", "start": 0, "end": 51, "score": 0.93}]},
  "latency_ms": 24.1, "n_windows": 1, "p99_exceeded": false, "error": null
}
```

`clauses` keeps the original response shape. `spans` adds character offsets and confidence. Also available: `GET /clauses`, `GET /health` and `GET /metrics`. Configuration (all optional) is listed in `.env.example`.

### Tests

```bash
pip install -r requirements-dev.txt
pytest && ruff check .
```

The tests build a tiny random BERT token classifier with the project's label set. They run inference, sliding windows, INT8 quantization, the NF4 load workaround and the FastAPI app end-to-end on CPU in a few seconds, with no model download.

---

## Project structure

```
cuad-extractor/
├── src/
│   ├── config.py              # clause types, label map, window size, paths, MLflow store
│   ├── spans.py               # BIO labelling/decoding, span merge, span & token F1 (pure Python)
│   ├── metrics.py             # bootstrap/Wilson CIs, latency summary, calibration (ECE)
│   ├── inference.py           # ClauseExtractor: sliding windows, one pass, scores, int8/nf4 loading
│   ├── training/
│   │   ├── prepare_data.py    # CUAD download, multi-span labels, negatives, contract-level splits, lock
│   │   ├── train.py           # fine-tuning + grid search, span-level val selection, MLflow
│   │   └── quantize.py        # INT8 (CPU) and NF4 (GPU) trade-off report
│   ├── eval/
│   │   ├── evaluate.py        # per-clause F1 + CIs, negatives, calibration, adversarial, val/test gap
│   │   ├── calibration.py     # reliability table from an eval report
│   │   └── monitor.py         # production drift + online labelling
│   └── serving/
│       ├── serve.py           # FastAPI app
│       └── load_test.py       # async load test
├── tests/                     # pytest suite (tiny local model, no downloads)
├── notebooks/kaggle_train.ipynb
├── .github/workflows/ci.yml
├── MODEL_CARD.md
└── requirements.txt / requirements-dev.txt / pyproject.toml
```

---

## Known issues & debugging notes

- **NaN loss from epoch 1:** AdamW's default `eps=1e-8` is too small for DeBERTa-v3's gradient-disentangled embedding sharing. Fixed with `eps=1e-6`. Batches with non-finite loss or gradients are skipped and counted in MLflow.
- **Collapse to all-`O`:** fixed with class-weighted loss (see point 8 above for the over-correction and its ablation).
- **4-bit NF4 load crash — root cause found.** It's not a long-standing upstream bug. It's a version interaction: `accelerate` ≥ 0.33 assumes 4-bit bitsandbytes models can be moved (true since bitsandbytes 0.43.2), so for a single-GPU device map it ends with `model.to(device)`. The pinned `transformers==4.44.2` still rejects `.to()` on *every* bitsandbytes model, which is exactly the error in the notebook. `src/inference.allow_noop_to_for_bnb` permits only no-op moves during loading; real moves still raise. This is unit-tested on CPU, but the end-to-end NF4 path still needs a GPU run to confirm. Meanwhile INT8 dynamic quantization gives a working CPU option.
- **`transformers` pin:** kept at 4.44.2 because that's the version the reported run was trained with. Upgrading would also remove the need for the NF4 workaround above.

## Stack

| Component | Technology |
|---|---|
| Base model | microsoft/deberta-v3-base (override with `CUAD_BASE_MODEL`) |
| Fine-tuning | HuggingFace Transformers 4.44.2 + PyTorch |
| Quantization | PyTorch dynamic INT8 (CPU); bitsandbytes NF4 (GPU) |
| Experiment tracking | MLflow (SQLite store in `experiments/`) |
| Serving | FastAPI + uvicorn |
| Dataset | CUAD v1 via `dvgodoy/CUAD_v1_Contract_Understanding_{clause_classification,PDF}` |
| CI | GitHub Actions: ruff + pytest |
